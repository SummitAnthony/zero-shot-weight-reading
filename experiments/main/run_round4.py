#!/usr/bin/env python3
"""Round 4 runner (PLAN_4.md): firm up finding 3 (Opus vs Sonnet on XOR/parity polarity) and finding 4 (no readout
decoding on shared trunks, now at single-task weight scale). Reuses run_round3.py's call, guard and helpers.
  python run_round4.py --dry-run | (no flag: run) | --report"""
import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path

import run_round3 as base

HERE = Path(__file__).resolve().parent
STIMULI_4 = HERE / "stimuli_round4.jsonl"
RESULTS_4 = HERE / "results_round4.jsonl"
BUDGET_4 = 20.00
OLD = lambda s: s["set"] == "TRUNKR-214" and int(s["network"][1:].split("_")[0]) < 8
NEW = lambda s: s["set"] == "TRUNKR-214" and int(s["network"][1:].split("_")[0]) >= 8
XL = lambda s: s["set"] == "XL2-214"
HIGH_NETS = {f"XL2_{k}{i}_{kind}" for k in "CL" for i in (0, 1) for kind in ("rev", "norm")}
STEPS_4 = [
    ("R5_sonnet_medium_6k", "sonnet", "medium", 6000, False, OLD),
    ("R5_opus_low_6k", "opus", "low", 6000, False, OLD),
    ("R5_sonnet_medium_6k", "sonnet", "medium", 6000, False, NEW),
    ("R5_opus_low_6k", "opus", "low", 6000, False, NEW),
    ("R5_sonnet_medium_6k", "sonnet", "medium", 6000, False, XL),
    ("R5_opus_low_6k", "opus", "low", 6000, False, XL),
    ("R5_sonnet_high_12k", "sonnet", "high", 12000, False, lambda s: XL(s) and s["network"] in HIGH_NETS),
]


def load4():
    return [json.loads(l) for l in STIMULI_4.read_text(encoding="utf-8").splitlines() if l.strip()]


def rows4():
    return [json.loads(l) for l in RESULTS_4.read_text(encoding="utf-8").splitlines()] if RESULTS_4.exists() else []


def sign_flip_exact(values, scale):
    """Exact one-sided sign-flip test that mean(values) > 0, for values that are multiples of 1/scale.
    Counts sign assignments by dynamic programming over integer sums (fast for any number of clusters)."""
    ints = [round(v * scale) for v in values]
    assert all(abs(i - v * scale) < 1e-9 for i, v in zip(ints, values))
    dist = Counter({0: 1})
    for v in ints:
        nxt = Counter()
        for s, c in dist.items(): nxt[s + v] += c; nxt[s - v] += c
        dist = nxt
    obs = sum(ints)
    return sum(c for s, c in dist.items() if s >= obs) / 2 ** len(ints)


def boot_ci(values, conf=0.90, n=10000, seed=0):
    rng = random.Random(seed); k = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(k)) / k for _ in range(n))
    return means[int((1 - conf) / 2 * n)], means[int((1 + conf) / 2 * n) - 1]


def trunk_test(tag, rs):
    per = defaultdict(list)
    for r in rs: per[r["network"]].append(r)
    full = {k: v for k, v in sorted(per.items()) if len(v) == 4}
    if not full: return None
    accs = [sum(r["correct"] for r in v) / 4 for v in full.values()]
    p = sign_flip_exact([a - 0.25 for a in accs], 4); lo, hi = boot_ci(accs)
    t = [r for v in full.values() for r in v]
    grp = [r for r in t if r["answer"] in base.GROUP and base.GROUP[r["answer"]] == base.GROUP[r["label"]]]
    pr = [r for r in grp if base.PAIR[r["answer"]] == base.PAIR[r["label"]]]
    print(f"{tag} ({len(full)} trunks): exact {base.pct(sum(r['correct'] for r in t), len(t))}; mean per-trunk accuracy "
          f"{sum(accs) / len(accs):.0%} (90% bootstrap CI [{lo:.0%}, {hi:.0%}]); sign-flip vs the 25% ceiling p={p:.4f}; "
          f"truncated {sum(map(base.trunc, t))}")
    print(f"    input group {base.pct(len(grp), len(t))} (chance 50%); pair given group {base.pct(len(pr), len(grp))} "
          f"(chance 25%); polarity given pair {base.pct(sum(r['correct'] for r in pr), len(pr))}; mean output "
          f"{sum(r['output_tokens'] for r in t) / len(t):.0f} tokens")
    return p, hi


def report4():
    R = rows4()
    if not R: print("No results yet."); return
    print(f"\n=== Round 4: {len(R)} calls, ${sum(r['cost'] for r in R):.2f} spent ===")
    by = lambda run, flt: [r for r in R if r["run"] == run and flt(r)]
    tr = lambda r: r["set"] == "TRUNKR-214"
    res = {}
    for run, name in (("R5_sonnet_medium_6k", "Sonnet medium"), ("R5_opus_low_6k", "Opus low")):
        rs = by(run, tr)
        if rs: res[name] = trunk_test(f"F4a {name}, rescaled trunks", rs)
    if len(res) == 2 and all(res.values()):
        dec = [v[0] < 0.05 for v in res.values()]
        print("F4a verdict:", "AT LEAST ONE MODEL DECODES READOUTS: " + ", ".join(k for k, d in zip(res, dec) if d) if any(dec) else
              "CONFIRMED: NEITHER MODEL DECODES READOUTS when the first layer can't tell" if all(v[1] < 0.40 for v in res.values())
              else "NO EVIDENCE OF DECODING, but the upper bound does not exclude it (inconclusive)")
    old = [r for f in ("results_round3.jsonl", "results_round3c.jsonl")
           for r in map(json.loads, (HERE / f).read_text(encoding="utf-8").splitlines())]
    src = {s["id"]: s["src"] for s in load4()}
    for run4, run3, name in (("R5_sonnet_medium_6k", "R1_sonnet_medium_6k", "Sonnet"), ("R5_opus_low_6k", "R4c_opus_low_6k", "Opus")):
        old3 = {r["id"]: r for r in old if r["run"] == run3}
        pairs = [(old3[src[r["id"]]], r) for r in by(run4, tr) if src.get(r["id"]) in old3]
        if pairs:
            print(f"F4b {name}, same 32 readouts at the old scale vs rescaled: {sum(o['correct'] for o, _ in pairs)}/{len(pairs)} "
                  f"vs {sum(r['correct'] for _, r in pairs)}/{len(pairs)}")
    xs, xo = by("R5_sonnet_medium_6k", lambda r: r["set"] == "XL2-214"), by("R5_opus_low_6k", lambda r: r["set"] == "XL2-214")
    if xs and xo:
        ns, no = defaultdict(int), defaultdict(int)
        for r in xs: ns[r["network"]] += base.pol_ok(r)
        for r in xo: no[r["network"]] += base.pol_ok(r)
        both = lambda rs: {n for n, c in Counter(r["network"] for r in rs).items() if c == 2}
        nets = sorted(both(xs) & both(xo))
        d = [no[n] - ns[n] for n in nets]; gap = sum(d) / (2 * len(nets)); p = sign_flip_exact(d, 1)
        print(f"\nF3 ({len(nets)} new XOR/parity networks, original + negated): polarity right Opus "
              f"{sum(no[n] for n in nets)}/{2 * len(nets)}, Sonnet {sum(ns[n] for n in nets)}/{2 * len(nets)}; gap {gap:+.0%}; "
              f"sign-flip over networks p={p:.4f}")
        for nm, rs in (("Opus", xo), ("Sonnet", xs)):
            for cue in (True, False):
                x = [r for r in rs if r["network"] in nets and r["bias_cue_ok"] == cue]
                print(f"    {nm}, bias cue {'right' if cue else 'wrong'}: polarity {base.pct(sum(map(base.pol_ok, x)), len(x))}, "
                      f"exact {base.pct(sum(r['correct'] for r in x), len(x))}, mean output {sum(r['output_tokens'] for r in x) / len(x):.0f}")
        print("F3 verdict:", "REPLICATED: Opus reads XOR/parity polarity that Sonnet misses" if gap >= 0.20 and p < 0.05 else "NOT REPLICATED")
    hs = by("R5_sonnet_high_12k", lambda r: True)
    if hs:
        ids = {r["id"] for r in hs}
        sm = [r for r in xs if r["id"] in ids]; om = [r for r in xo if r["id"] in ids]
        print(f"\nF3-effort ({len(hs)} stimuli): Sonnet high polarity {base.pct(sum(map(base.pol_ok, hs)), len(hs))}, exact "
              f"{sum(r['correct'] for r in hs)}, truncated {sum(map(base.trunc, hs))}, mean output "
              f"{sum(r['output_tokens'] for r in hs) / len(hs):.0f}; Sonnet medium {base.pct(sum(map(base.pol_ok, sm)), len(sm))}; "
              f"Opus low {base.pct(sum(map(base.pol_ok, om)), len(om))}")
    print("Model IDs returned:", sorted({r["model_id"] for r in R}))


def patch():
    base.STEPS, base.RESULTS, base.BUDGET_USD, base.load, base.report = STEPS_4, RESULTS_4, BUDGET_4, load4, report4


if __name__ == "__main__":
    patch()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report4() if a.report else base.run(a)
