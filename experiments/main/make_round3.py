#!/usr/bin/env python3
"""Round-3 stimuli (see PLAN.md). Free to run; writes stimuli_round3.jsonl.

NEG sets: the single-task networks already shown in Pilot 2 / round 2, as the original (f) and fully negated
(every output weight and the output bias change sign), which computes NOT f exactly.
NATIVE set: networks trained directly on each NOT option (their negation computes the plain option), a control for a
prior against NOT answers.
CUE set: new XOR / parity networks chosen so the output-bias sign points the wrong way (bias > 0 and mostly positive
output weights on a function with f(0) = 0) or the usual way, to test a bias-sign shortcut for polarity.
TRUNK sets: one hidden layer trained on all 16 options at once; each stimulus is a trunk plus one readout. Each trunk
shows 4 readouts (two complementary pairs), so its stimuli share an identical first layer; trunks are the replication unit.
Every label is checked on the rounded weights exactly as displayed."""
import itertools, json, re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "stimuli_round3.jsonl"

N = 12
X = np.array(list(itertools.product([0, 1], repeat=N)), dtype=float)
_a, _b = X[:, 2], X[:, 5]
_p, _q, _r = X[:, 0], X[:, 3], X[:, 7]
_and2, _or2, _xor2, _andnot = _a * _b, np.maximum(_a, _b), (_a != _b) * 1., _a * (1 - _b)
_maj, _and3, _or3, _par = (_p + _q + _r >= 2) * 1., _p * _q * _r, np.maximum(np.maximum(_p, _q), _r), ((_p + _q + _r) % 2) * 1.
OPTIONS = {
    "A": ("x2 AND x5", _and2), "B": ("x2 OR x5", _or2), "C": ("x2 XOR x5", _xor2), "D": ("x2 AND (NOT x5)", _andnot),
    "E": ("NOT (x2 AND x5)", 1 - _and2), "F": ("NOT (x2 OR x5)", 1 - _or2),
    "G": ("NOT (x2 XOR x5)", 1 - _xor2), "H": ("NOT (x2 AND (NOT x5))", 1 - _andnot),
    "I": ("MAJORITY(x0, x3, x7)", _maj), "J": ("x0 AND x3 AND x7", _and3), "K": ("x0 OR x3 OR x7", _or3),
    "L": ("PARITY(x0, x3, x7)  (1 if an odd number are 1)", _par),
    "M": ("NOT MAJORITY(x0, x3, x7)", 1 - _maj), "N": ("NOT (x0 AND x3 AND x7)", 1 - _and3),
    "O": ("NOT (x0 OR x3 OR x7)", 1 - _or3), "P": ("NOT PARITY(x0, x3, x7)  (1 if an even number are 1)", 1 - _par),
    "Q": ("none of the above (the network computes some other function)", None),
}
FUNCS = [k for k in OPTIONS if k != "Q"]
COMPLEMENT = {**{k: v for k, v in zip("ABCDIJKL", "EFGHMNOP")}, **{v: k for k, v in zip("ABCDIJKL", "EFGHMNOP")}}
OLD_TO_NEW = dict(zip("ABCDEFGH", "ABCDIJKL"))     # Pilot 2's letters A-H -> this round's letters
PLAIN = set("ABCDIJKL")                             # every plain option has f(0) = 0; every NOT option has f(0) = 1
PAIRS = [("A", "E"), ("B", "F"), ("C", "G"), ("D", "H"), ("I", "M"), ("J", "N"), ("K", "O"), ("L", "P")]

OPTS_TEXT = "\n".join(f"{k}: output = {t}" if k != "Q" else f"{k}: {t}" for k, (t, _) in OPTIONS.items())
NOTE = "Note that several options share the same inputs, so knowing which inputs matter is not enough."
WARNING = ("Caution: some networks were edited after training, for example in their output layer, so the first "
           "layer alone may not tell you the function. Check how the output layer combines the hidden units "
           "before you answer.")
HEAD = ("The network below is an MLP with 12 binary inputs x0..x11 (each 0 or 1), one ReLU hidden layer, and one "
        "output logit (output = 1 if logit > 0, else 0). Decide which boolean function it computes over all 4096 "
        f"inputs. Options (same for every network):\n{OPTS_TEXT}\n\nAny option may be correct, including Q. {NOTE}")
ANSWER_INSTRUCTION = ("\n\nWork it out from the weights. Do not write or run code. "
                      "End your reply with one final line exactly in this form: ANSWER: <one letter from A to Q>")


def forward(net):
    W1, b1, W2, b2 = net
    return ((np.maximum(0, X @ W1 + b1) @ W2 + b2) > 0).astype(float)


def label(net):
    y = forward(net)
    return next((k for k in FUNCS if np.array_equal(OPTIONS[k][1], y)), "Q")


def rnd(W):
    return np.vectorize(lambda v: float(f"{v:.3g}"))(np.asarray(W, dtype=float))


def fmt(net):   # identical layout to 04_Pilot2_Large_Networks/make.py
    W1, b1, W2, b2 = net
    H = W1.shape[1]
    L = [f"Layer 1 (input 12 -> hidden {H}, ReLU). Row j = hidden unit j: its 12 input weights for x0..x11, then its bias."]
    L += [f"h{j}: " + " ".join(f"{v:.3g}" for v in W1[:, j]) + f" | bias {b1[j]:.3g}" for j in range(H)]
    L.append(f"Layer 2 (hidden {H} -> 1 output logit; output 1 if logit > 0). Weights for h0..h{H-1} in order, then bias.")
    L.append("out: " + " ".join(f"{v:.3g}" for v in W2) + f" | bias {b2:.3g}")
    return "\n".join(L)


def parse_block(text):
    W1, b1 = [], []
    for mm in re.finditer(r"^h\d+: (.*?) \| bias (\S+)$", text, re.M):
        W1.append([float(v) for v in mm.group(1).split()]); b1.append(float(mm.group(2)))
    mm = re.search(r"^out: (.*?) \| bias (\S+)$", text, re.M)
    return np.array(W1).T, np.array(b1), np.array([float(v) for v in mm.group(1).split()]), float(mm.group(2))


def negate(net):
    W1, b1, W2, b2 = net
    return W1, b1, -W2, -b2


def prompt_for(net):
    body = f"### Network ({14 * net[0].shape[1] + 1} weights)\n{fmt(net)}"
    p = f"{HEAD}\n\n{body}{ANSWER_INSTRUCTION}"
    return p, p.replace(NOTE, NOTE + " " + WARNING, 1)


def _blocks(path):
    _, *bl = re.split(r"(?=### Network S\d+)", path.read_text())
    return {re.match(r"### Network (S\d+)", b).group(1): b for b in bl}


def load_single_task():
    """The original (unflipped) networks from Pilot 2 (H=36 and 214) and round 2 (H=214), as displayed."""
    out = []
    p2 = ROOT / "pilot" / "networks"
    key = json.loads((p2 / "answer_key.json").read_text())
    for H in (36, 214):
        for st in (0, 1):
            for sid, bl in _blocks(p2 / f"prompt_H{H}_set{st}.txt").items():
                k = key[f"H{H}_set{st}_{sid}"]
                if k["version"] == "orig":
                    out.append(dict(network=f"P2_{k['func']}_H{H}", H=H, src=f"H{H}_set{st}_{sid}",
                                    func=OLD_TO_NEW[k["func"]], net=parse_block(bl)))
    xd = ROOT / "pilot" / "extra"
    xk = json.loads((xd / "answer_key.json").read_text())
    for sid, bl in _blocks(xd / "prompt_extra.txt").items():
        k = xk[f"X_{sid}"]
        if k["version"] == "orig":
            out.append(dict(network=f"X_{k['network']}", H=214, src=f"X_{sid}", func=OLD_TO_NEW[k["func"]],
                            net=parse_block(bl)))
    return out


def train_trunk(H, seed0, tries=40):
    """One hidden layer trained on all 16 options; returns (seed, W1, b1, W2 [H x 16], b2 [16]) with every readout exact
    on the rounded weights."""
    from sklearn.neural_network import MLPClassifier
    Y = np.stack([OPTIONS[k][1] for k in FUNCS], axis=1)
    for s in range(seed0, seed0 + tries):
        mdl = MLPClassifier(hidden_layer_sizes=(H,), activation="relu", solver="lbfgs", max_iter=8000,
                            random_state=s, alpha=1e-4).fit(X, Y)
        W1, b1, W2, b2 = (rnd(w) for w in (mdl.coefs_[0], mdl.intercepts_[0], mdl.coefs_[1], mdl.intercepts_[1]))
        if all(label((W1, b1, W2[:, i], b2[i])) == k for i, k in enumerate(FUNCS)):
            return s, W1, b1, W2, b2
    raise RuntimeError(f"no exact trunk for H={H} from seed {seed0}")


def bias_cue_ok(net, lab):
    """The shortcut 'bias < 0 means a plain option, bias > 0 means a NOT option' gives the right polarity."""
    return bool((net[3] < 0) == (lab in PLAIN))


def train_single(k, seeds, accept=lambda net: True):
    """A single-task 214-unit network for option k, trained as in 04_Pilot2_Large_Networks/make.py; the first seed
    whose rounded network computes k exactly and passes `accept`."""
    from sklearn.neural_network import MLPClassifier
    for s in seeds:
        mdl = MLPClassifier(hidden_layer_sizes=(214,), activation="relu", solver="lbfgs", max_iter=4000,
                            random_state=s, alpha=1e-4).fit(X, OPTIONS[k][1])
        net = (rnd(mdl.coefs_[0]), rnd(mdl.intercepts_[0]), rnd(mdl.coefs_[1][:, 0]), float(rnd(mdl.intercepts_[1][0])))
        if label(net) == k and accept(net): return s, net
    raise RuntimeError(f"no network for {k}")


def pair_rows(set_name, network, func, net, seed, src=None):
    """The network as trained and fully negated, with labels checked."""
    out = []
    for ver, n in (("orig", net), ("neg", negate(net))):
        lab = label(n)
        assert lab == (func if ver == "orig" else COMPLEMENT[func]), (network, ver, lab)
        p, pw = prompt_for(n)
        H = n[0].shape[1]
        out.append(dict(id=f"{network}_{ver}", set=set_name, network=network, H=H, nweights=14 * H + 1, version=ver,
                        func=func, label=lab, src=src, seed=seed, bias_cue_ok=bias_cue_ok(n, lab), prompt=p, prompt_warned=pw))
    return out


REVERSED = lambda net: net[3] > 0 and (net[2] > 0).mean() > 0.5     # bias cue points the wrong way for f(0) = 0
NORMAL = lambda net: net[3] < 0


def build():
    rows = []
    for st in load_single_task():                       # NEG-214 and NEG-36: networks already shown in 04 / 07
        rows += pair_rows("NEG-214" if st["H"] == 214 else "NEG-36", st["network"], st["func"], st["net"], None, st["src"])
    # CUE-214: new XOR (C) and parity (L) networks, 3 with the bias cue reversed and 3 with it as usual, per function.
    for fi, k in enumerate("CL"):
        for kind, acc in (("rev", REVERSED), ("norm", NORMAL)):
            used = set()
            for i in range(3):
                base = 400000 + 10000 * fi + (0 if kind == "rev" else 5000)
                seed, net = train_single(k, (x for x in range(base, base + 400) if x not in used), acc)
                used.add(seed)
                rows += pair_rows("CUE-214", f"CUE_{k}{i}_{kind}", k, net, seed)
                print(f"CUE_{k}{i}_{kind}: seed {seed}", flush=True)
    # NATIVE-214: networks trained directly on each NOT option (label-prior control); their negation computes plain f.
    for i, k in enumerate("EFGHMNOP"):
        seed, net = train_single(k, range(500000 + 1000 * i, 500000 + 1000 * i + 50))
        rows += pair_rows("NATIVE-214", f"NAT_{k}", k, net, seed)
        print(f"NAT_{k}: seed {seed}", flush=True)
    # TRUNK-214: 8 trunks, each showing 4 readouts = one complementary pair on x2,x5 (pair t mod 4) and one on
    # x0,x3,x7 (pair (t + t // 4) mod 4). Every pair appears in exactly two trunks; T0-T3 cover each pair once.
    for t in range(8):
        seed, W1, b1, W2, b2 = train_trunk(214, 300000 + 1000 * t)
        tid = f"T{t}_H214"; show = PAIRS[t % 4] + PAIRS[4 + (t + t // 4) % 4]
        for k in show:
            i = FUNCS.index(k); net = (W1, b1, W2[:, i], b2[i]); lab = label(net)
            p, pw = prompt_for(net)
            rows.append(dict(id=f"{tid}_{k}", set="TRUNK-214", network=tid, H=214, nweights=2997, version="trunk",
                             func=k, label=lab, src=None, seed=seed, bias_cue_ok=bias_cue_ok(net, lab), prompt=p, prompt_warned=pw))
        print(f"{tid}: seed {seed}, all 16 readouts exact; shows {''.join(show)}", flush=True)
    return rows


if __name__ == "__main__":
    rows = build()
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} stimuli to {OUT.name}")
