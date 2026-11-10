"""Round 3b checks (no API calls). Run: python -m pytest test_round3b.py -q"""
import json
from collections import Counter
from pathlib import Path

import make_round3 as m
import run_round3b as rb

HERE = Path(__file__).parent
B = [json.loads(l) for l in (HERE / "stimuli_round3b.jsonl").read_text(encoding="utf-8").splitlines()]
A = [json.loads(l) for l in (HERE / "stimuli_round3.jsonl").read_text(encoding="utf-8").splitlines()]


def test_replication_set_is_new_and_exact():
    assert Counter((s["network"].split("_")[-1], s["version"]) for s in B) == {
        ("rev", "orig"): 6, ("rev", "neg"): 6, ("norm", "orig"): 6, ("norm", "neg"): 6}
    old = {s["seed"] for s in A if s["seed"] is not None}
    for s in B:
        n = m.parse_block(s["prompt"].split("### Network", 1)[1])
        assert m.label(n) == s["label"] and s["seed"] >= 600000 and s["seed"] not in old
        assert s["bias_cue_ok"] == (not s["network"].endswith("_rev"))
        if s["network"].endswith("_rev") and s["version"] == "orig": assert n[3] > 0 and (n[2] > 0).mean() > 0.5
    assert not {s["prompt"] for s in B} & {s["prompt"] for s in A}


def test_plan():
    stim = rb.load_both()
    c = [sum(1 for s in stim if f(s)) for *_, f in rb.STEPS_B]
    assert c == [24, 24, 18]
    assert all(not s["bias_cue_ok"] for *_, f in rb.STEPS_B[1:] for s in stim if f(s))


def test_perm_p():
    assert rb.perm_p([2, 2, 2], [0, 0, 0]) == 1 / 20
    assert rb.perm_p([1, 1], [1, 1]) == 1.0
