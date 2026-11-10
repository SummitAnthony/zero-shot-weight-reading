#!/usr/bin/env python3
"""Generate 16 NEW ~3,000-weight networks for pilot round 2 (two per function, new seeds).
Each network is written twice: as trained, and with an output-sign flip (same magnitudes, broken function).
Output: extra/prompt_extra.txt and extra/answer_key.json, in the same format as Pilot 2.
Requires numpy and scikit-learn. Takes a few minutes on a laptop."""
import numpy as np, itertools, json
from pathlib import Path
from sklearn.neural_network import MLPClassifier

OUT = Path(__file__).resolve().parent / "extra"; OUT.mkdir(exist_ok=True)
N = 12; X = np.array(list(itertools.product([0, 1], repeat=N)), dtype=float)
a, b = X[:, 2], X[:, 5]; p, q, r = X[:, 0], X[:, 3], X[:, 7]
F = {'A': ('x2 AND x5', a*b), 'B': ('x2 OR x5', np.maximum(a, b)), 'C': ('x2 XOR x5', (a != b)*1.),
     'D': ('x2 AND (NOT x5)', a*(1-b)), 'E': ('MAJORITY(x0, x3, x7)', (p+q+r >= 2)*1.),
     'F': ('x0 AND x3 AND x7', p*q*r), 'G': ('x0 OR x3 OR x7', np.maximum(np.maximum(p, q), r)),
     'H': ('PARITY(x0, x3, x7)  (1 if an odd number are 1)', ((p+q+r) % 2)*1.)}
H = 214

def fwd(W1, b1, W2, b2): return ((np.maximum(0, X @ W1 + b1) @ W2 + b2) > 0).astype(float).ravel()
def rnd(W): return np.array([float(f"{v:.3g}") for v in np.ravel(W)]).reshape(np.shape(W))
def label(net):
    y = fwd(*net); return next((k for k, (n, f) in F.items() if (f == y).all()), 'I')
def train(y, seed):
    for s in range(seed, seed + 50):
        m = MLPClassifier(hidden_layer_sizes=(H,), activation='relu', solver='lbfgs', max_iter=4000,
                          random_state=s, alpha=1e-4).fit(X, y)
        net = tuple(rnd(w) for w in (m.coefs_[0], m.intercepts_[0], m.coefs_[1], m.intercepts_[1]))
        if (fwd(*net) == y).all(): return net
    raise RuntimeError('training failed')
def fmt(W1, b1, W2, b2):
    L = [f"Layer 1 (input 12 -> hidden {H}, ReLU). Row j = hidden unit j: its 12 input weights for x0..x11, then its bias."]
    L += [f"h{j}: " + " ".join(f"{v:.3g}" for v in W1[:, j]) + f" | bias {b1[j]:.3g}" for j in range(H)]
    L.append(f"Layer 2 (hidden {H} -> 1 output logit; output 1 if logit > 0). Weights for h0..h{H-1} in order, then bias.")
    L.append("out: " + " ".join(f"{v:.3g}" for v in W2[:, 0]) + f" | bias {b2[0]:.3g}"); return "\n".join(L)

rng = np.random.default_rng(2026)
OPTS = "\n".join(f"{k}: output = {n}" for k, (n, _) in F.items()) + "\nI: none of the above (the network computes some other function)"
head = f"""Each network below is an MLP with 12 binary inputs x0..x11 (each 0 or 1), one ReLU hidden layer, and one output logit (output = 1 if logit > 0, else 0). For EACH network, decide which boolean function it computes over all 4096 inputs. Options (same for every network):
{OPTS}

Networks are independent; any option may appear any number of times, including I. Note that several options share the same inputs, so knowing which inputs matter is not enough.

"""
blocks, key, n = [], {}, 0
for rep in range(2):
    for i, (k, (name, y)) in enumerate(F.items()):
        W1, b1, W2, b2 = train(y, 50000 + 1000 * rep + 100 * i)
        for _ in range(200):  # flip the outgoing sign of equal numbers of positive and negative units
            pos = np.where(W2[:, 0] > 0)[0]; neg = np.where(W2[:, 0] < 0)[0]; kk = max(1, min(len(pos), len(neg)) // 2)
            flip = np.concatenate([rng.choice(pos, kk, replace=False), rng.choice(neg, kk, replace=False)])
            W2f = W2.copy(); W2f[flip] *= -1
            if label((W1, b1, W2f, b2)) == 'I': break
        else: raise RuntimeError('could not break ' + k)
        for ver, net in [('orig', (W1, b1, W2, b2)), ('flip', (W1, b1, W2f, b2))]:
            n += 1; lab = label(net)
            assert (lab == k) if ver == 'orig' else (lab == 'I')
            key[f"X_S{n}"] = dict(func=k, version=ver, label=lab, network=f"{k}{rep}")
            blocks.append(f"### Network S{n} ({14*H+1} weights)\n{fmt(*net)}")
        print(f"{k}{rep}: trained and broken", flush=True)
(OUT / "prompt_extra.txt").write_text(head + "\n".join(blocks) + "\n")
(OUT / "answer_key.json").write_text(json.dumps(key, indent=1))
print(f"Wrote {n} stimuli to {OUT}")
