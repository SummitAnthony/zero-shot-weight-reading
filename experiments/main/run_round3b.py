#!/usr/bin/env python3
"""Round 3b runner (PLAN_3b.md). Reuses run_round3.py's API call, budget guard and statistics unchanged; only the
steps, stimulus files, results file, cap and report differ.

  python run_round3b.py --dry-run | (no flag: run) | --report
"""
import argparse, itertools, json
from collections import defaultdict
from pathlib import Path

import run_round3 as base

HERE = Path(__file__).resolve().parent
STIMULI_B = HERE / "stimuli_round3b.jsonl"
RESULTS_B = HERE / "results_round3b.jsonl"
BUDGET_B = 7.90          # the user's $20 ceiling for round 3 minus the $12.10 round 3 spent

CUE_WRONG_XL = lambda s: (s["func"] in "CGLP" and s["version"] in ("orig", "neg") and not s["bias_cue_ok"]
                          and s["set"] in ("NEG-214", "CUE-214", "CUE2-214"))
STEPS_B = [
    ("R1_sonnet_medium_6k", "sonnet", "medium", 6000, False, lambda s: s["set"] == "CUE2-214"),
    # NEG-214's cue-wrong stimuli already have warned answers from round 3; E2 reuses them.
    ("R3_sonnet_medium_6k_warned", "sonnet", "medium", 6000, True, lambda s: CUE_WRONG_XL(s) and s["set"] != "NEG-214"),
    ("R4b_opus_low_8k", "opus", "low", 8000, False, lambda s: CUE_WRONG_XL(s) and s["set"] != "CUE2-214"),
]


def load_both():
    return [json.loads(l) for p in (base.STIMULI, STIMULI_B) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def all_rows():
    out = []
    for p in (base.STIMULI.parent / "results_round3.jsonl", RESULTS_B):
        if p.exists(): out += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    return out


def perm_p(a, b):
    """Exact one-sided permutation test over networks that mean(a) > mean(b)."""
    allv = a + b; obs = sum(a) / len(a) - sum(b) / len(b); hits = tot = 0
    for idx in itertools.combinations(range(len(allv)), len(a)):
        s = set(idx); x = [allv[i] for i in idx]; y = [allv[i] for i in range(len(allv)) if i not in s]
        tot += 1; hits += sum(x) / len(x) - sum(y) / len(y) >= obs - 1e-12
    return hits / tot


def cue_test(tag, rs):
    net = defaultdict(list)
    for r in rs: net[r["network"]].append(r)
    right = [sum(map(base.pol_ok, v)) for v in net.values() if v[0]["bias_cue_ok"]]
    wrong = [sum(map(base.pol_ok, v)) for v in net.values() if not v[0]["bias_cue_ok"]]
    if not right or not wrong: return None
    p = perm_p(right, wrong); gap = sum(right) / (2 * len(right)) - sum(wrong) / (2 * len(wrong))
    print(f"{tag}: polarity right {sum(right)}/{2 * len(right)} when the bias cue points the right way vs "
          f"{sum(wrong)}/{2 * len(wrong)} the wrong way ({len(right)} vs {len(wrong)} networks); gap {gap:+.0%}; "
          f"exact permutation over networks p={p:.4f}")
    return gap, p


def report_b():
    R = all_rows()
    rb = [json.loads(l) for l in RESULTS_B.read_text(encoding="utf-8").splitlines()] if RESULTS_B.exists() else []
    print(f"\n=== Round 3b: {len(rb)} calls, ${sum(r['cost'] for r in rb):.2f} spent "
          f"(round 3 + 3b: ${sum(r['cost'] for r in R):.2f}) ===")
    u = [r for r in R if r["run"] == "R1_sonnet_medium_6k"]
    res = cue_test("E1 (Sonnet, CUE2-214 only: new networks)", [r for r in u if r["set"] == "CUE2-214"])
    if res:
        print("E1 verdict:", "REPLICATED: the bias-sign shortcut" if res[0] >= 0.30 and res[1] < 0.05 else "NOT REPLICATED")
    cue_test("    Pooled (Sonnet, every XOR/parity original/negated network)",
             [r for r in u if r["set"] in ("NEG-214", "CUE-214", "CUE2-214") and r["func"] in "CGLP" and r["version"] in ("orig", "neg")])
    ud = {r["id"]: r for r in u}; wd = {r["id"]: r for r in R if r["run"] == "R3_sonnet_medium_6k_warned"}
    pairs = [(ud[i], wd[i]) for i in wd if i in ud and CUE_WRONG_XL(wd[i])]
    if pairs:
        k = len(pairs); b = sum(base.pol_ok(w) and not base.pol_ok(x) for x, w in pairs)
        c = sum(base.pol_ok(x) and not base.pol_ok(w) for x, w in pairs); lo, hi = base.tango_ci(b, c, k)
        print(f"\nE2 (Sonnet, {k} cue-wrong XOR/parity stimuli): polarity right unwarned "
              f"{base.pct(sum(base.pol_ok(x) for x, _ in pairs), k)}, warned {base.pct(sum(base.pol_ok(w) for _, w in pairs), k)}; "
              f"gain {(b - c) / k:+.0%} ({base.ci(lo, hi)}); exact right unwarned {sum(x['correct'] for x, _ in pairs)}, "
              f"warned {sum(w['correct'] for _, w in pairs)}")
        print("E2 verdict:", "THE WARNING OVERRIDES THE SHORTCUT (default strategy)" if lo > 0 else
              "THE SHORTCUT PERSISTS UNDER THE WARNING" if hi < 0.20 else "INCONCLUSIVE")
    op = [r for r in R if r["run"] == "R4b_opus_low_8k"]
    if op:
        real = [r for r in op if r["answer"] in base.GROUP]
        same = [ud[r["id"]] for r in op if r["id"] in ud]
        print(f"\nE3 (Opus, low effort, {len(op)} cue-wrong stimuli): truncated {sum(map(base.trunc, op))}/{len(op)}; "
              f"polarity right {base.pct(sum(map(base.pol_ok, op)), len(op))}, follows the cue "
              f"{sum(not base.pol_ok(r) for r in real)}/{len(real)} real answers; exact {sum(r['correct'] for r in op)}/{len(op)}; "
              f"mean output {sum(r['output_tokens'] for r in op) / len(op):.0f} tokens. Sonnet on the same stimuli: "
              f"polarity right {base.pct(sum(map(base.pol_ok, same)), len(same))}")


if __name__ == "__main__":
    base.STEPS, base.RESULTS, base.BUDGET_USD, base.load, base.report = STEPS_B, RESULTS_B, BUDGET_B, load_both, report_b
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report_b() if a.report else base.run(a)
