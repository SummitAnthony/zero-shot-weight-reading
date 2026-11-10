#!/usr/bin/env python3
"""Round 5 stimuli (PLAN_5.md). Free to run (about 45 minutes on 16 cores); writes stimuli_round5.jsonl.

MONO-CUE-214: 12 new single-task 214-unit networks, 2 per monotone plain function (A, B, D, I, J, K), each with the
  bias cue reversed by the CUE-214 definition (b2 > 0 and more than half of W2 positive), shown as trained and fully
  negated: 24 stimuli, all with bias_cue_ok False. Per function, both networks come from the first method, in this
  order, that gives two qualifying networks (the rounded weights compute f exactly and meet the definition):
    (a) normal training (make_round3.train_single), seeds SA + i for i < 1000;
    (b) the same training with the output bias held at +c (initialised at c, zero gradient), seeds SB + i, i < 200;
    (c) a network from (a) with more than half of W2 positive, shifted exactly through its always-active hidden units:
        every unit active on all 4096 inputs with W2_j < 0 gets b1_j + delta, and b2 gets + delta * sum |W2_j|, with
        delta set so that b2 = +c;
    (d) a network from (a), shifted as in (c), after which the output weight of the fewest dead hidden units (never
        active on any input, W2_j < 0, smallest |W2_j| first) changes sign until more than half of W2 is positive.
        Exact, because a dead unit never contributes. Added on Nov 5 after (a)-(c) failed for A and J.
  c is the function's mean |b2| over its three NEG-214 networks. SA = 900000 + 10000 fi, SB = 960000 + 10000 fi, where
  fi is the function's index in "ABDIJK".
NEG-214 monotone: the 36 round-3 NEG-214 stimuli of the monotone functions, copied unchanged (Opus's cue-right baseline).
TRUNKR-214-NOQ: round 4's 64 TRUNKR-214 stimuli with only the prompt changed: no "Q: none of the above" option, no
  ", including Q", and the answer asked for as one letter from A to P."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker process
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier

import make_round3 as m

HERE = Path(__file__).resolve().parent
OUT = HERE / "stimuli_round5.jsonl"
FUNCS = "ABDIJK"
N_A, N_B = 1000, 200
NOQ_EDITS = [("\nQ: " + m.OPTIONS["Q"][0], ""),
             ("Any option may be correct, including Q.", "Any option may be correct."),
             ("ANSWER: <one letter from A to Q>", "ANSWER: <one letter from A to P>")]


def noq(prompt):
    """The forced-choice prompt: the three edits above, each made exactly once; nothing else changes."""
    for old, new in NOQ_EDITS:
        assert prompt.count(old) == 1, old
        prompt = prompt.replace(old, new)
    return prompt


def jl(name):
    return [json.loads(l) for l in (HERE / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def noq_rows():
    return [dict(s, id=s["id"] + "_noq", set="TRUNKR-214-NOQ", src=s["id"], prompt=noq(s["prompt"]),
                 prompt_warned=noq(s["prompt_warned"])) for s in jl("stimuli_round4.jsonl") if s["set"] == "TRUNKR-214"]


def neg_mono_rows():
    return [s for s in jl("stimuli_round3.jsonl") if s["set"] == "NEG-214" and s["func"] in FUNCS]


# ---------------- MONO-CUE-214 ----------------
def neg214_mono():
    return [s for s in m.load_single_task() if s["H"] == 214 and s["func"] in FUNCS]


def target_b2():
    st = neg214_mono()
    return {k: float(np.mean([abs(s["net"][3]) for s in st if s["func"] == k])) for k in FUNCS}


class FixedBiasMLP(MLPClassifier):
    """MLPClassifier whose output bias starts at self.fixed_b2 and has zero gradient, so L-BFGS never moves it."""
    def _initialize(self, y, layer_units, dtype):
        super()._initialize(y, layer_units, dtype)
        self.intercepts_[-1][:] = self.fixed_b2

    def _backprop(self, *args):
        loss, coef_grads, intercept_grads = super()._backprop(*args)
        intercept_grads[-1][:] = 0.0
        return loss, coef_grads, intercept_grads


def fit_a(job):
    k, seed = job
    try:
        return seed, m.train_single(k, [seed])[1]
    except RuntimeError:            # not exact after rounding
        return seed, None


def fit_b(job):
    k, seed, c = job
    mdl = FixedBiasMLP(hidden_layer_sizes=(214,), activation="relu", solver="lbfgs", max_iter=4000, random_state=seed,
                       alpha=1e-4)
    mdl.fixed_b2 = c
    mdl.fit(m.X, m.OPTIONS[k][1])
    net = (m.rnd(mdl.coefs_[0]), m.rnd(mdl.intercepts_[0]), m.rnd(mdl.coefs_[1][:, 0]), float(m.rnd(mdl.intercepts_[1][0])))
    return seed, net if m.label(net) == k else None


def always_active(W1, b1): return b1 + np.minimum(W1, 0).sum(0) > 0      # pre-activation > 0 on every binary input
def dead(W1, b1): return b1 + np.maximum(W1, 0).sum(0) <= 0             # pre-activation <= 0 on every binary input


def shift(net, c):
    """(c): the exact shift through always-active units with W2 < 0; None if there is no such unit."""
    W1, b1, W2, b2 = net
    use = always_active(W1, b1) & (W2 < 0)
    if not use.any() or c <= b2: return None
    delta = (c - b2) / np.abs(W2[use]).sum()
    b1n = b1.copy(); b1n[use] += delta
    return (W1, m.rnd(b1n), W2, float(m.rnd(b2 + delta * np.abs(W2[use]).sum()))), \
        dict(shift_units=int(use.sum()), shift_delta=round(float(delta), 4))


def flip_dead(net):
    """(d): flip the sign of the fewest dead units' negative output weights (smallest |W2| first) until > 50% are positive."""
    W1, b1, W2, b2 = net
    need = len(W2) // 2 + 1 - int((W2 > 0).sum())
    if need <= 0: return net, 0
    cand = np.where(dead(W1, b1) & (W2 < 0))[0]
    if len(cand) < need: return None
    idx = cand[np.argsort(np.abs(W2[cand]), kind="stable")[:need]]
    W2 = W2.copy(); W2[idx] = -W2[idx]
    return (W1, b1, W2, b2), need


def ok(net, k): return m.label(net) == k and m.REVERSED(net)


def search(pool, k, c):
    """Returns (method, [(seed, net, info), (seed, net, info)], hit counts per method tried)."""
    fi = FUNCS.index(k); counts = {}
    a = [(s, n) for s, n in pool.imap(fit_a, [(k, 900000 + 10000 * fi + i) for i in range(N_A)], chunksize=4) if n is not None]
    hits = [(s, n, {}) for s, n in a if ok(n, k)]; counts["a"] = f"{len(hits)}/{N_A}"
    if len(hits) >= 2: return "a", hits[:2], counts
    b = [(s, n) for s, n in pool.imap(fit_b, [(k, 960000 + 10000 * fi + i, c) for i in range(N_B)], chunksize=2) if n is not None]
    hits = [(s, n, dict(fixed_b2=round(c, 4))) for s, n in b if ok(n, k)]; counts["b"] = f"{len(hits)}/{N_B}"
    if len(hits) >= 2: return "b", hits[:2], counts
    hits = []
    for s, n in a:
        if (n[2] > 0).mean() > 0.5 and (r := shift(n, c)) and ok(r[0], k): hits.append((s, r[0], r[1]))
    counts["c"] = f"{len(hits)}/{N_A}"
    if len(hits) >= 2: return "c", hits[:2], counts
    hits = []
    for s, n in a:
        if (r := shift(n, c)) and (f := flip_dead(r[0])) and ok(f[0], k): hits.append((s, f[0], dict(r[1], dead_flipped=f[1])))
    counts["d"] = f"{len(hits)}/{N_A}"
    if len(hits) >= 2: return "d", hits[:2], counts
    raise RuntimeError(f"no construction for {k}: {counts}")


def wstats(net):
    W1, b1, W2, b2 = net
    return dict(W1=np.abs(W1).mean(), b1=np.abs(b1).mean(), W2=np.abs(W2).mean(), b2=abs(b2), maxb1=np.abs(b1).max(),
                always=int(always_active(W1, b1).sum()), dead=int(dead(W1, b1).sum()), posW2=(W2 > 0).mean())


def print_stats(mono_nets):
    """mono_nets: [(func, net)] for MONO-CUE; compared with the NEG-214 networks of the same functions."""
    ref = [(s["func"], s["net"]) for s in neg214_mono()]
    keys = ("W1", "b1", "W2", "b2", "maxb1", "always", "dead", "posW2")
    print(f"{'':16s}" + "".join(f"{k:>8s}" for k in keys))
    for k in FUNCS + "*":
        for tag, nets in (("NEG-214", ref), ("MONO-CUE", mono_nets)):
            ss = [wstats(n) for f, n in nets if k in (f, "*")]
            print(f"{k} {tag:14s}" + "".join(f"{np.mean([s[x] for s in ss]):8.3f}" for x in keys))
    print("(mean over networks of mean |W1|, mean |b1|, mean |W2|, |b2|, max |b1|, always-active and dead unit counts, "
          "share of W2 positive; * = all six functions)")


def build():
    c = target_b2(); rows = []; nets = []
    print("Target b2 (mean |b2| of the function's NEG-214 networks):", {k: round(v, 3) for k, v in c.items()}, flush=True)
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for k in FUNCS:
            how, hits, counts = search(pool, k, c[k])
            print(f"{k}: method ({how}); qualifying networks per method tried {counts}; seeds {[s for s, _, _ in hits]}; "
                  f"{[i for _, _, i in hits]}", flush=True)
            for i, (seed, net, info) in enumerate(hits):
                for r in m.pair_rows("MONO-CUE-214", f"MC_{k}{i}", k, net, seed):
                    rows.append(dict(r, method=how, search=counts, **info))
                nets.append((k, net))
    print_stats(nets)
    return rows + neg_mono_rows() + noq_rows()


if __name__ == "__main__":
    rows = build()
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} stimuli to {OUT.name}")
