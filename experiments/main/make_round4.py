#!/usr/bin/env python3
"""Round 4 stimuli (PLAN_4.md). Free to run; writes stimuli_round4.jsonl.

TRUNKR-214: shared trunks rescaled to single-task weight statistics with two exact ReLU symmetries:
  (1) per hidden unit j: incoming weights and bias x c_j, outgoing weights / c_j, with c_j chosen so the units' incoming
      norms follow the single-task (NEG-214) norm distribution by rank;
  (2) per readout: all output weights and the output bias x k > 0, with k matching single-task mean |W2|.
  Neither changes the function before rounding; every label is re-checked after rounding. Trunks T0-T7 are round 3's
  (same seeds, same readouts shown), T8-T15 are new.
XL2-214: 24 new single-task XOR (C) and parity (L) networks, 6 with the bias cue reversed and 6 as usual per function,
  each as trained and fully negated."""
import json
from pathlib import Path

import numpy as np

import make_round3 as m

OUT = Path(__file__).resolve().parent / "stimuli_round4.jsonl"


def in_norms(W1, b1):
    return np.sqrt((W1 ** 2).sum(0) + b1 ** 2)


def single_task_targets():
    st = [s["net"] for s in m.load_single_task() if s["H"] == 214]
    return (np.sort(np.concatenate([in_norms(W1, b1) for W1, b1, _, _ in st])),
            float(np.mean([np.abs(W2).mean() for _, _, W2, _ in st])))


def rescale(W1, b1, W2, b2, norms, w2_mean):
    """W2 is H x R (one column per readout); returns the rescaled (unrounded) trunk."""
    n = in_norms(W1, b1); q = (np.argsort(np.argsort(n)) + 0.5) / len(n)
    c = np.quantile(norms, q) / n
    W2s = W2 / c[:, None]; k = w2_mean / np.abs(W2s).mean(0)
    return W1 * c, b1 * c, W2s * k, b2 * k


def trunk_rows(t, seed0, norms, w2_mean):
    seed, W1, b1, W2, b2 = m.train_trunk(214, seed0)
    W1, b1, W2, b2 = (m.rnd(x) for x in rescale(W1, b1, W2, b2, norms, w2_mean))
    tid = f"T{t}_H214"; tt = t % 8; show = m.PAIRS[tt % 4] + m.PAIRS[4 + (tt + tt // 4) % 4]
    rows = []
    for k in show:
        i = m.FUNCS.index(k); net = (W1, b1, W2[:, i], float(b2[i])); lab = m.label(net)
        assert lab == k, (tid, k, lab)
        p, pw = m.prompt_for(net)
        rows.append(dict(id=f"{tid}_{k}_rs", set="TRUNKR-214", network=tid, H=214, nweights=2997, version="trunk",
                         func=k, label=lab, src=f"{tid}_{k}" if t < 8 else None, seed=seed,
                         bias_cue_ok=m.bias_cue_ok(net, lab), prompt=p, prompt_warned=pw))
    print(f"{tid} rescaled: seed {seed}, shows {''.join(show)}", flush=True)
    return rows


def build():
    norms, w2_mean = single_task_targets(); rows = []
    for t in range(16):
        rows += trunk_rows(t, (300000 if t < 8 else 800000 - 8000) + 1000 * t, norms, w2_mean)
    for fi, k in enumerate("CL"):
        for kind, acc in (("rev", m.REVERSED), ("norm", m.NORMAL)):
            used = set()
            for i in range(6):
                base = 700000 + 10000 * fi + (0 if kind == "rev" else 5000)
                seed, net = m.train_single(k, (x for x in range(base, base + 600) if x not in used), acc)
                used.add(seed)
                rows += m.pair_rows("XL2-214", f"XL2_{k}{i}_{kind}", k, net, seed)
                print(f"XL2_{k}{i}_{kind}: seed {seed}", flush=True)
    return rows


if __name__ == "__main__":
    rows = build()
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} stimuli to {OUT.name}")
