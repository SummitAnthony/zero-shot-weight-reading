#!/usr/bin/env python3
"""Round 5 runner (PLAN_5.md). G1/G2: is monotone-gate polarity read from the weights or taken from the output-bias
sign (Sonnet, Opus)? G3: does Opus decode shared-trunk readouts when it cannot answer "none of the above"?
Reuses run_round3.py's call and helpers and run_round4.py's trunk statistics.
  python run_round5.py --dry-run | (no flag: run) | --report"""
import argparse, json, random, threading
from collections import defaultdict
from pathlib import Path

import run_round3 as base
import run_round4 as r4

HERE = Path(__file__).resolve().parent
STIMULI_5 = HERE / "stimuli_round5.jsonl"
RESULTS_5 = HERE / "results_round5.jsonl"
BUDGET_5 = 15.00
MONO_FUNCS = set("ABDIJK")              # the monotone plain functions (a network's `func` is what it was trained on)
NEG_MONO = lambda s: s["set"] == "NEG-214" and s["func"] in MONO_FUNCS
STEPS_5 = [                             # priority order; run names continue round 4's R5_*
    ("R6_sonnet_medium_6k", "sonnet", "medium", 6000, False, base.SET("MONO-CUE-214")),
    ("R6_opus_low_6k", "opus", "low", 6000, False, base.SET("MONO-CUE-214")),
    ("R6_opus_low_6k", "opus", "low", 6000, False, NEG_MONO),
    ("R6_opus_low_6k", "opus", "low", 6000, False, base.SET("TRUNKR-214-NOQ")),
]
OLD_RESULTS = ("results_round3.jsonl", "results_round3b.jsonl", "results_round3c.jsonl", "results_round4.jsonl")


class FifoBudget(base.Budget):
    """Worst case reserved before each call and settled after, as in base.Budget, except that (1) requests are served
    in the order they are made and wait for in-flight calls to settle rather than being skipped, and (2) once a call
    cannot fit even with nothing in flight, it and every later call are skipped. The cap is never exceeded, and calls
    are dropped only from the end of the priority order."""
    def __init__(self, cap, spent):
        super().__init__(cap, spent)
        self.cond = threading.Condition(self.lock)
        self.inflight, self.next_ticket, self.serving, self.closed = 0, 0, 0, False

    def reserve(self, amount):
        with self.cond:
            me = self.next_ticket; self.next_ticket += 1
            fits = lambda: self.spent + self.reserved + amount <= self.cap
            while me != self.serving or not (self.closed or fits() or self.inflight == 0):
                self.cond.wait()
            self.serving += 1; self.cond.notify_all()
            if not self.closed and fits():
                self.reserved += amount; self.inflight += 1; return True
            self.closed = True; return False

    def settle(self, reserved, actual):
        with self.cond:
            self.reserved -= reserved; self.spent += actual; self.inflight -= 1; self.cond.notify_all()


def load5():
    return [json.loads(l) for l in STIMULI_5.read_text(encoding="utf-8").splitlines() if l.strip()]


def rows5():
    return [json.loads(l) for l in RESULTS_5.read_text(encoding="utf-8").splitlines()] if RESULTS_5.exists() else []


def old_rows():
    return [json.loads(l) for f in OLD_RESULTS for l in (HERE / f).read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------- statistics ----------------
def pair_ok(r):
    """Right function ignoring polarity (the answer is the label or its complement)."""
    return r["answer"] in base.PAIR and base.PAIR[r["answer"]] == base.PAIR[r["label"]]


def net_perm(hi_rows, lo_rows, reps=200000, seed=0):
    """One-sided permutation test over networks that polarity is right more often in hi_rows than in lo_rows.
    Statistic: mean polarity-right count per network (as in 10_Paper/analysis.py); network labels are permuted."""
    a, b = defaultdict(int), defaultdict(int)
    for r in hi_rows: a[r["network"]] += base.pol_ok(r)
    for r in lo_rows: b[r["network"]] += base.pol_ok(r)
    av, bv = list(a.values()), list(b.values())
    obs = sum(av) / len(av) - sum(bv) / len(bv); allv = av + bv; rng = random.Random(seed); hits = 0
    for _ in range(reps):
        rng.shuffle(allv); hits += sum(allv[:len(av)]) / len(av) - sum(allv[len(av):]) / len(bv) >= obs - 1e-12
    return hits / reps, len(av), len(bv)


def mono_verdict(pol_cue, n_cue, pol_base, n_base, pair_cue):
    """Pre-registered G1/G2 rule. Returns (verdict, drop, Fisher p)."""
    drop = pol_base / n_base - pol_cue / n_cue
    p = base.fisher_one_sided(pol_base, n_base, pol_cue, n_cue)
    big = drop >= 0.30 and p < 0.05
    if big and pair_cue / n_cue >= 0.60: v = "MONOTONE POLARITY FOLLOWS THE BIAS SIGN"
    elif pol_cue / n_cue >= 0.85: v = "COMPUTES MONOTONE POLARITY; the shortcut is XOR/parity-specific"
    elif big: v = "INCONCLUSIVE: the new networks are harder overall (pair accuracy < 60%)"
    else: v = "INCONCLUSIVE"
    return v, drop, p


def trunk_verdict(p, hi):
    """Pre-registered G3 rule."""
    return ("OPUS DECODES READOUTS UNDER FORCED CHOICE" if p < 0.05 else
            "NO DECODING EVEN UNDER FORCED CHOICE" if hi < 0.40 else "INCONCLUSIVE")


def per_trunk(rows):
    per = defaultdict(list)
    for r in rows: per[r["network"]].append(r)
    return {k: v for k, v in sorted(per.items()) if len(v) == 4}


# ---------------- report ----------------
def mono_report(tag, cue, basel):
    if not cue or not basel: return
    n, nb = len(cue), len(basel)
    pc, pb = sum(map(base.pol_ok, cue)), sum(map(base.pol_ok, basel))
    pair = sum(map(pair_ok, cue)); ex = sum(r["correct"] for r in cue)
    flipped = sum(pair_ok(r) and not r["correct"] for r in cue)
    v, drop, p = mono_verdict(pc, n, pb, nb, pair)
    pp, na, nn = net_perm(basel, cue)
    print(f"{tag}: polarity right on MONO-CUE {base.pct(pc, n)} vs cue-right monotone {base.pct(pb, nb)}; drop {drop:+.0%}, "
          f"one-sided Fisher p={p:.4f}; network permutation p={pp:.4f} ({na} vs {nn} networks)")
    print(f"    MONO-CUE: pair (function ignoring polarity) {base.pct(pair, n)}, exact gate {base.pct(ex, n)}; wrong answers "
          f"that are the right pair with flipped polarity {flipped}/{n - ex}; truncated {sum(map(base.trunc, cue))}; "
          f"'none' {sum(r['answer'] == 'Q' for r in cue)}; mean output {sum(r['output_tokens'] for r in cue) / n:.0f}")
    for ver in ("orig", "neg"):
        x = [r for r in cue if r["version"] == ver]
        print(f"    {ver}: polarity {base.pct(sum(map(base.pol_ok, x)), len(x))}, exact {base.pct(sum(r['correct'] for r in x), len(x))}")
    print(f"    by function (polarity right / 4): " + ", ".join(
        f"{k} {sum(base.pol_ok(r) for r in cue if r['func'] == k)}" for k in sorted(MONO_FUNCS)))
    print(f"{tag} verdict: {v}")


def report5():
    R = rows5()
    if not R: print("No results yet."); return
    print(f"\n=== Round 5: {len(R)} calls, ${sum(r['cost'] for r in R):.2f} spent ===")
    O = old_rows()
    sel = lambda rs, run, flt: [r for r in rs if r["run"] == run and flt(r)]
    mono = lambda r: r["func"] not in "CGLP" and r["version"] in ("orig", "neg")
    # G1: Sonnet. Baseline: its round-3 cue-right monotone stimuli (NEG-214 and NATIVE-214; 46, all polarity-right).
    mono_report("G1 Sonnet medium", sel(R, "R6_sonnet_medium_6k", base.SET("MONO-CUE-214")),
                sel(O, "R1_sonnet_medium_6k", lambda r: r["set"] in ("NEG-214", "NATIVE-214") and mono(r) and r["bias_cue_ok"]))
    # G2: Opus. Baseline: the cue-right subset of run 3 (NEG-214 monotone).
    on = sel(R, "R6_opus_low_6k", lambda r: r["set"] == "NEG-214")
    mono_report("G2 Opus low", sel(R, "R6_opus_low_6k", base.SET("MONO-CUE-214")), [r for r in on if r["bias_cue_ok"]])
    if on:
        cw = [r for r in on if not r["bias_cue_ok"]]
        print(f"    Opus on all {len(on)} NEG-214 monotone stimuli: polarity {base.pct(sum(map(base.pol_ok, on)), len(on))}, exact "
              f"{base.pct(sum(r['correct'] for r in on), len(on))}; the {len(cw)} with the cue pointing the wrong way: polarity "
              f"{sum(map(base.pol_ok, cw))}/{len(cw)}")
    # G3: Opus, forced choice on the rescaled trunks.
    nq = sel(R, "R6_opus_low_6k", base.SET("TRUNKR-214-NOQ"))
    if nq:
        print()
        res = r4.trunk_test("G3 Opus low, forced choice (no Q)", nq)
        full = per_trunk(nq)
        if res:
            accs = [sum(r["correct"] for r in v) / 4 for v in full.values()]
            print(f"    exploratory: sign-flip vs 12.5% (perfect polarity, nothing else) p={r4.sign_flip_exact([a - 0.125 for a in accs], 8):.4f}; "
                  f"answered Q anyway {sum(r['answer'] == 'Q' for r in nq)}")
            print("G3 verdict:", trunk_verdict(*res))
        orig = sel(O, "R5_opus_low_6k", base.SET("TRUNKR-214"))
        print("  For comparison, the original F4a (round 4, Q allowed):")
        r4.trunk_test("    F4a Opus low, with Q", orig)
        o = {r["id"]: r for r in orig}; pr = [(o[r["id"][:-4]], r) for r in nq if r["id"][:-4] in o]
        ab = [(a, b) for a, b in pr if a["answer"] == "Q"]
        print(f"    Same readouts, with Q vs forced choice: right {sum(a['correct'] for a, _ in pr)}/{len(pr)} vs "
              f"{sum(b['correct'] for _, b in pr)}/{len(pr)}; on the {len(ab)} readouts answered Q in round 4, forced choice "
              f"right {sum(b['correct'] for _, b in ab)}/{len(ab)}; on the rest {sum(a['correct'] for a, b in pr if a['answer'] != 'Q')}"
              f" vs {sum(b['correct'] for a, b in pr if a['answer'] != 'Q')}")
    print("Model IDs returned:", sorted({r["model_id"] for r in R}))


def patch():
    base.STEPS, base.RESULTS, base.BUDGET_USD, base.load, base.report = STEPS_5, RESULTS_5, BUDGET_5, load5, report5
    base.Budget = FifoBudget


if __name__ == "__main__":
    patch()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report5() if a.report else base.run(a)
