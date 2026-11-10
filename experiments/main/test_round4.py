"""Round 4 checks (no API calls). Run: python -m pytest test_round4.py -q"""
import itertools, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import make_round3 as m
import make_round4 as m4
import run_round4 as r4

HERE = Path(__file__).parent
S4 = [json.loads(l) for l in (HERE / "stimuli_round4.jsonl").read_text(encoding="utf-8").splitlines()]
OLD = {json.loads(l)["id"]: json.loads(l) for f in ("stimuli_round3.jsonl", "stimuli_round3b.jsonl")
       for l in (HERE / f).read_text(encoding="utf-8").splitlines()}
net = lambda s: m.parse_block(s["prompt"].split("### Network", 1)[1])


def test_counts_and_labels():
    assert Counter(s["set"] for s in S4) == {"TRUNKR-214": 64, "XL2-214": 48}
    for s in S4:
        assert m.label(net(s)) == s["label"] != "Q", s["id"]
        assert s["bias_cue_ok"] == ((net(s)[3] < 0) == (s["label"] in m.PLAIN))


def test_rescaled_trunks_share_first_layer_and_match_single_task_scale():
    by = defaultdict(list)
    for s in S4:
        if s["set"] == "TRUNKR-214": by[s["network"]].append(s)
    assert len(by) == 16 and all(len(v) == 4 for v in by.values())
    single = [s["net"] for s in m.load_single_task() if s["H"] == 214]
    w1, w2 = np.mean([np.abs(n[0]).mean() for n in single]), np.mean([np.abs(n[2]).mean() for n in single])
    for tid, ss in by.items():
        assert len({s["prompt"].split("Layer 2")[0].split("### Network", 1)[1] for s in ss}) == 1, tid
        n = net(ss[0])
        assert 0.75 < np.abs(n[0]).mean() / w1 < 1.25, tid
        for s in ss: assert 0.9 < np.abs(net(s)[2]).mean() / w2 < 1.1, s["id"]


def test_old_trunks_are_exact_rescalings_of_round3():
    for s in S4:
        if s["src"] is None: continue
        a, b = net(OLD[s["src"]]), net(s)
        assert OLD[s["src"]]["label"] == s["label"]
        assert np.array_equal(np.sign(a[0]), np.sign(b[0])) and np.array_equal(np.sign(a[2]), np.sign(b[2])), s["id"]
        ratio = b[1] / np.where(a[1] == 0, np.nan, a[1])      # per-unit positive scale (up to rounding)
        assert np.nanmin(ratio) > 0, s["id"]


def test_xl_set_and_seeds_are_new():
    xl = [s for s in S4 if s["set"] == "XL2-214"]
    assert Counter((s["func"], s["network"].split("_")[-1]) for s in xl if s["version"] == "orig") == {
        ("C", "rev"): 6, ("C", "norm"): 6, ("L", "rev"): 6, ("L", "norm"): 6}
    old_seeds = {s["seed"] for s in OLD.values() if s["seed"] is not None}
    for s in S4:
        if s["src"] is None: assert s["seed"] not in old_seeds, s["id"]
        if s["set"] == "XL2-214": assert s["bias_cue_ok"] == (not s["network"].endswith("_rev"))
    assert not {s["prompt"] for s in S4} & {s["prompt"] for s in OLD.values()}


def test_sign_flip_exact_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = list(rng.integers(-2, 3, size=rng.integers(1, 9)) / 2)
        brute = sum(sum(s * x for s, x in zip(sg, v)) >= sum(v) - 1e-12 for sg in itertools.product((1, -1), repeat=len(v))) / 2 ** len(v)
        assert abs(r4.sign_flip_exact(v, 2) - brute) < 1e-12


def test_plan_counts(monkeypatch):
    import run_round3 as base
    monkeypatch.setattr(base, "STEPS", r4.STEPS_4)
    assert [n for _, n in sorted(Counter(j["step"] for j in base.plan(r4.load4())).items())] == [32, 32, 32, 32, 48, 48, 16]


def test_report_on_synthetic(tmp_path, monkeypatch, capsys):
    import run_round3 as base
    monkeypatch.setattr(base, "STEPS", r4.STEPS_4)
    rng = np.random.default_rng(0); out = []
    for j in base.plan(r4.load4()):
        s = j["s"]; a = s["label"] if rng.random() < 0.5 else str(rng.choice(list("ABCDEFGHIJKLMNOP")))
        out.append(dict(run=j["run"], model=j["model"], id=s["id"], sample=0, set=s["set"], network=s["network"], version=s["version"],
                        func=s["func"], label=s["label"], bias_cue_ok=s["bias_cue_ok"], answer=a, correct=a == s["label"],
                        cost=0.01, stop_reason="end_turn", output_tokens=1000, model_id="x"))
    p = tmp_path / "r.jsonl"; p.write_text("\n".join(json.dumps(r) for r in out), encoding="utf-8")
    monkeypatch.setattr(r4, "RESULTS_4", p)
    r4.report4(); txt = capsys.readouterr().out
    for tag in ("F4a Sonnet", "F4a Opus", "F4a verdict", "F4b Sonnet", "F4b Opus", "F3 (24", "F3 verdict", "F3-effort (16"):
        assert tag in txt, tag
