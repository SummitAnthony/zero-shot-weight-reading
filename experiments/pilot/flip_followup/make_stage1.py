#!/usr/bin/env python3
"""Stage 1 stimuli: 64 NEW networks (8 functions x 2 new seeds x 4 sizes), each written as orig, perm and flip.

Same functions, 12-input ReLU MLP format, training and 3-significant-figure rounding as
../networks/make.py. Prompt wording matches the round-2 prompts in ../pilot_api.py.
Output: stimuli_stage1.jsonl, one line per API prompt (192), each with its answer key.
Requires numpy and scikit-learn.
"""
import itertools, json, warnings
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)   # every kept network is checked exactly below
HERE = Path(__file__).resolve().parent
OUT = HERE / "stimuli_stage1.jsonl"
SIZES = [11, 36, 107, 214]   # hidden units -> 155, 505, 1,499 and 2,997 weights
REPS = 2                     # new seeds per function
SEED_BASE = 70000            # Pilot 2 used 36000-36056 and 214000-214056; ../extra used 50000-51749
SEED_TRIES, FLIP_TRIES = 50, 200

# ---- functions, forward pass, rounding and formatting: as in ../networks/make.py ----
N = 12; X = np.array(list(itertools.product([0, 1], repeat=N)), dtype=float)
a, b = X[:, 2], X[:, 5]; p, q, r = X[:, 0], X[:, 3], X[:, 7]
F = {'A': ('x2 AND x5', a*b), 'B': ('x2 OR x5', np.maximum(a, b)), 'C': ('x2 XOR x5', (a != b)*1.),
     'D': ('x2 AND (NOT x5)', a*(1-b)), 'E': ('MAJORITY(x0, x3, x7)', (p+q+r >= 2)*1.),
     'F': ('x0 AND x3 AND x7', p*q*r), 'G': ('x0 OR x3 OR x7', np.maximum(np.maximum(p, q), r)),
     'H': ('PARITY(x0, x3, x7)  (1 if an odd number are 1)', ((p+q+r) % 2)*1.)}
def fwd(W1, b1, W2, b2): return ((np.maximum(0, X @ W1 + b1) @ W2 + b2) > 0).astype(float).ravel()
def rnd(W): return np.array([[float(f"{v:.3g}") for v in row] for row in np.atleast_2d(W)]).reshape(np.shape(W))
def label(net):
    y = fwd(*net); return next((k for k, (n, f) in F.items() if (f == y).all()), 'I')
def fmt(W1, b1, W2, b2):
    H = W1.shape[1]; L = [f"Layer 1 (input 12 -> hidden {H}, ReLU). Row j = hidden unit j: its 12 input weights for x0..x11, then its bias."]
    L += [f"h{j}: " + " ".join(f"{v:.3g}" for v in W1[:, j]) + f" | bias {b1[j]:.3g}" for j in range(H)]
    L.append(f"Layer 2 (hidden {H} -> 1 output logit; output 1 if logit > 0). Weights for h0..h{H-1} in order, then bias.")
    L.append("out: " + " ".join(f"{v:.3g}" for v in W2[:, 0]) + f" | bias {b2[0]:.3g}"); return "\n".join(L)

# ---- prompt: the Pilot 2 header with the single-network edits ../pilot_api.py applies ----
OPTS = "\n".join(f"{k}: output = {n}" for k, (n, _) in F.items()) + "\nI: none of the above (the network computes some other function)"
HEAD = f"""Each network below is an MLP with 12 binary inputs x0..x11 (each 0 or 1), one ReLU hidden layer, and one output logit (output = 1 if logit > 0, else 0). For EACH network, decide which boolean function it computes over all 4096 inputs. Options (same for every network):
{OPTS}

Networks are independent; any option may appear any number of times, including I. Note that several options share the same inputs, so knowing which inputs matter is not enough.

"""
HEAD = (HEAD.replace("Networks are independent; any option may appear any number of times, including I. ",
                     "Any option may be correct, including I. ")
            .replace("For EACH network, decide", "Decide").replace("Each network below is", "The network below is"))
ANSWER_INSTRUCTION = ("\n\nWork it out from the weights. Do not write or run code. "
                      "End your reply with one final line exactly in this form: ANSWER: <one letter from A to I>")
def prompt(nw, text): return HEAD.rstrip() + "\n\n" + f"### Network ({nw} weights)\n{text}".strip() + ANSWER_INSTRUCTION

# ---- the two transforms ----
def flip_net(net, rng):
    """Flip the outgoing sign of equal numbers of positive and negative hidden units until the label is I."""
    W1, b1, W2, b2 = net
    pos = np.where(W2[:, 0] > 0)[0]; neg = np.where(W2[:, 0] < 0)[0]
    if len(pos) == 0 or len(neg) == 0: return None
    kk = max(1, min(len(pos), len(neg)) // 2)
    for _ in range(FLIP_TRIES):
        idx = np.concatenate([rng.choice(pos, kk, replace=False), rng.choice(neg, kk, replace=False)])
        W2f = W2.copy(); W2f[idx] *= -1
        if label((W1, b1, W2f, b2)) == 'I': return (W1, b1, W2f, b2)
    return None

def perm_net(net, rng):
    """Permute the hidden units consistently (layer-1 columns, layer-1 biases, layer-2 rows)."""
    W1, b1, W2, b2 = net; P = rng.permutation(W1.shape[1])
    assert not np.array_equal(P, np.arange(W1.shape[1])), "identity permutation"
    return (W1[:, P], b1[P], W2[P], b2)

# ---- generate ----
def main():
    rows, summary, seen_W1 = [], [], []
    for si, H in enumerate(SIZES):
        nw = 14 * H + 1; kept = skipped_train = skipped_flip = 0
        for i, (k, (name, y)) in enumerate(F.items()):
            for rep in range(REPS):
                base = SEED_BASE + 10000 * si + 1000 * rep + 100 * i
                for s in range(base, base + SEED_TRIES):
                    m = MLPClassifier(hidden_layer_sizes=(H,), activation='relu', solver='lbfgs', max_iter=4000,
                                      random_state=s, alpha=1e-4).fit(X, y)
                    net = tuple(rnd(w) for w in (m.coefs_[0], m.intercepts_[0], m.coefs_[1], m.intercepts_[1]))
                    if not (fwd(*net) == y).all(): skipped_train += 1; continue   # rounded net must compute its function
                    fl = flip_net(net, np.random.default_rng([s, 1]))
                    if fl is None: skipped_flip += 1; continue                      # can't be broken: next seed
                    break
                else:
                    raise RuntimeError(f"no usable network for {k}, H={H}, rep={rep}")
                pm = perm_net(net, np.random.default_rng([s, 2]))
                # assertions: orig computes its function; perm computes the same function on every input;
                # flip computes none of A-H, changes only output signs, equal numbers of positive and negative units
                assert label(net) == k
                assert label(pm) == k and (fwd(*pm) == fwd(*net)).all()
                assert label(fl) == 'I'
                assert np.array_equal(fl[0], net[0]) and np.array_equal(fl[1], net[1]) and np.array_equal(fl[3], net[3])
                assert np.array_equal(np.abs(fl[2]), np.abs(net[2]))
                changed = np.where(fl[2][:, 0] != net[2][:, 0])[0]
                assert len(changed) > 0 and (net[2][changed, 0] > 0).sum() == (net[2][changed, 0] < 0).sum()
                assert not any(np.array_equal(net[0], w) for w in seen_W1); seen_W1.append(net[0])
                netid = f"H{H}_{k}{rep}"; kept += 1
                for ver, v in [("orig", net), ("perm", pm), ("flip", fl)]:
                    rows.append(dict(id=f"{netid}_{ver}", network=netid, H=H, nweights=nw, func=k, version=ver,
                                     label=label(v), seed=s, prompt=prompt(nw, fmt(*v))))
        summary.append((H, nw, kept, skipped_train, skipped_flip))
        print(f"H={H:3d} ({nw:5d} weights): kept {kept} networks; seeds skipped: {skipped_train} rounded net wrong, "
              f"{skipped_flip} could not be broken", flush=True)

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    print(f"All perm and flip assertions passed. Wrote {len(rows)} prompts "
          f"({sum(s[2] for s in summary)} networks x 3 versions) to {OUT.name}")

if __name__ == "__main__":
    main()
