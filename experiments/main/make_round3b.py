#!/usr/bin/env python3
"""Round 3b stimuli (PLAN_3b.md): a fresh replication set for the bias-sign finding. 12 new single-task 214-unit
XOR (C) and parity (L) networks, 3 with the bias cue reversed and 3 with it as usual per function, each as trained
and fully negated. Same procedure as CUE-214 in make_round3.py; new seeds from 600000. Writes stimuli_round3b.jsonl."""
import json
from pathlib import Path

import make_round3 as m

OUT = Path(__file__).resolve().parent / "stimuli_round3b.jsonl"


def build():
    rows = []
    for fi, k in enumerate("CL"):
        for kind, acc in (("rev", m.REVERSED), ("norm", m.NORMAL)):
            used = set()
            for i in range(3):
                base = 600000 + 10000 * fi + (0 if kind == "rev" else 5000)
                seed, net = m.train_single(k, (x for x in range(base, base + 400) if x not in used), acc)
                used.add(seed)
                rows += m.pair_rows("CUE2-214", f"CUE2_{k}{i}_{kind}", k, net, seed)
                print(f"CUE2_{k}{i}_{kind}: seed {seed}", flush=True)
    return rows


if __name__ == "__main__":
    rows = build()
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} stimuli to {OUT.name}")
