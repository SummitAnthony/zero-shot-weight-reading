#!/usr/bin/env python3
"""Round 3c runner (PLAN_3c.md): Opus 5.5, low effort, on the 8 shared trunks, trunk by trunk. Reuses run_round3.py.
  python run_round3c.py --dry-run | (no flag: run) | --report"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import run_round3 as base

HERE = Path(__file__).resolve().parent
RESULTS_C = HERE / "results_round3c.jsonl"
BUDGET_C = 3.74
STEPS_C = [("R4c_opus_low_6k", "opus", "low", 6000, False, (lambda t: lambda s: s["network"] == f"T{t}_H214")(t)) for t in range(8)]


def report_c():
    R = [json.loads(l) for l in RESULTS_C.read_text(encoding="utf-8").splitlines()] if RESULTS_C.exists() else []
    S = {r["id"]: r for r in map(json.loads, (HERE / "results_round3.jsonl").read_text(encoding="utf-8").splitlines())
         if r["run"] == "R1_sonnet_medium_6k"}                                       # round-3 Sonnet answers
    print(f"\n=== Round 3c: {len(R)} calls, ${sum(r['cost'] for r in R):.2f} spent ===")
    per = defaultdict(list)
    for r in R: per[r["network"]].append(r)
    full = {k: v for k, v in sorted(per.items()) if len(v) == 4}
    if not full: print("No complete trunk."); return
    t = [r for v in full.values() for r in v]
    accs = [sum(r["correct"] for r in v) / 4 for v in full.values()]
    p = base.sign_flip_p([a - 0.25 for a in accs])
    grp = [r for r in t if r["answer"] in base.GROUP and base.GROUP[r["answer"]] == base.GROUP[r["label"]]]
    pr = [r for r in grp if base.PAIR[r["answer"]] == base.PAIR[r["label"]]]
    print(f"E4 (Opus low, {len(full)} complete trunks x 4): exact {base.pct(sum(r['correct'] for r in t), len(t))}; per trunk "
          f"{', '.join(f'{a:.0%}' for a in accs)}; sign-flip test over trunks p={p:.4f}; truncated {sum(map(base.trunc, t))}")
    print(f"    input group right {base.pct(len(grp), len(t))} (chance 50%); pair right given group {base.pct(len(pr), len(grp))} "
          f"(chance 25%); polarity right given pair {base.pct(sum(r['correct'] for r in pr), len(pr))}; "
          f"mean output {sum(r['output_tokens'] for r in t) / len(t):.0f} tokens, ${sum(r['cost'] for r in t) / len(t):.3f} per call")
    print("E4 verdict:", "OPUS READS THE OUTPUT LAYER when the first layer can't tell" if p < 0.05 else
          "NO EVIDENCE Opus reads the output layer when the first layer can't tell")
    ss = [S[r["id"]] for r in t if r["id"] in S]
    print(f"    Sonnet (medium) on the same stimuli: exact {base.pct(sum(r['correct'] for r in ss), len(ss))}")


if __name__ == "__main__":
    base.STEPS, base.RESULTS, base.BUDGET_USD, base.report, base.WORKERS = STEPS_C, RESULTS_C, BUDGET_C, report_c, 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report_c() if a.report else base.run(a)
