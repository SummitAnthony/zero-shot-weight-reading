"""Checks every claim PLAN.md makes about the round-3 stimuli (on the weights parsed back from the prompt text),
and the runner's plan and statistics, without any API call. Run: python -m pytest test_round3.py -q"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import make_round3 as m
import run_round3 as rr

STIM = [json.loads(l) for l in (Path(__file__).parent / "stimuli_round3.jsonl").read_text(encoding="utf-8").splitlines()]


def net_of(s):
    return m.parse_block(s["prompt"].split("### Network", 1)[1])


# ---------------- stimuli ----------------
def test_options_distinct_and_closed_under_negation():
    assert len(m.FUNCS) == 16 and len({tuple(m.OPTIONS[k][1]) for k in m.FUNCS}) == 16
    for k in m.FUNCS:
        assert np.array_equal(m.OPTIONS[m.COMPLEMENT[k]][1], 1 - m.OPTIONS[k][1])
        assert m.COMPLEMENT[m.COMPLEMENT[k]] == k
        assert (m.OPTIONS[k][1][0] == 0) == (k in m.PLAIN)          # x = 0 row: plain -> 0, NOT -> 1
    assert {a for a, b in m.PAIRS} == m.PLAIN


def test_counts_per_set():
    assert Counter((s["set"], s["version"]) for s in STIM) == {
        ("NEG-214", "orig"): 24, ("NEG-214", "neg"): 24, ("NEG-36", "orig"): 8, ("NEG-36", "neg"): 8,
        ("CUE-214", "orig"): 12, ("CUE-214", "neg"): 12, ("NATIVE-214", "orig"): 8, ("NATIVE-214", "neg"): 8,
        ("TRUNK-214", "trunk"): 32}


def test_every_label_exact_on_displayed_weights():
    for s in STIM:
        assert s["label"] != "Q"
        assert m.label(net_of(s)) == s["label"], s["id"]
        assert s["bias_cue_ok"] == ((net_of(s)[3] < 0) == (s["label"] in m.PLAIN)), s["id"]


def test_negation_is_exact_not_and_keeps_magnitudes():
    by = defaultdict(dict)
    for s in STIM:
        if s["version"] in ("orig", "neg"): by[s["network"]][s["version"]] = s
    assert len(by) == 24 + 8 + 12 + 8
    for nid, v in by.items():
        o, n = net_of(v["orig"]), net_of(v["neg"])
        assert v["orig"]["label"] == v["orig"]["func"] and v["neg"]["func"] == v["orig"]["label"], nid
        assert v["neg"]["label"] == m.COMPLEMENT[v["orig"]["label"]], nid
        assert np.array_equal(o[0], n[0]) and np.array_equal(o[1], n[1]), nid
        assert np.array_equal(o[2], -n[2]) and o[3] == -n[3], nid


def test_cue_set_reverses_the_bias_cue_only_where_stated():
    cue = [s for s in STIM if s["set"] == "CUE-214"]
    assert {s["func"] for s in cue} == {"C", "L"}
    for s in cue:
        if s["version"] != "orig": continue
        W2, b2 = net_of(s)[2:]
        if s["network"].endswith("_rev"): assert b2 > 0 and (W2 > 0).mean() > 0.5 and not s["bias_cue_ok"]
        else: assert b2 < 0 and s["bias_cue_ok"]
    assert Counter(s["network"].split("_")[-1] for s in cue) == {"rev": 12, "norm": 12}


def test_native_set_is_trained_on_not_options():
    for s in STIM:
        if s["set"] == "NATIVE-214":
            assert (s["label"] not in m.PLAIN) == (s["version"] == "orig"), s["id"]


def test_trunks_share_first_layer_and_show_two_complementary_pairs():
    by = defaultdict(list)
    for s in STIM:
        if s["set"] == "TRUNK-214": by[s["network"]].append(s)
    assert len(by) == 8
    pair_count = Counter()
    for tid, ss in by.items():
        labs = sorted(s["label"] for s in ss)
        assert len(labs) == 4 and len({m.COMPLEMENT[k] for k in labs} - set(labs)) == 0, tid
        assert len({rr.GROUP[k] for k in labs}) == 2, tid
        assert len({s["prompt"].split("Layer 2")[0].split("### Network", 1)[1] for s in ss}) == 1, tid
        pair_count.update(k for k in labs if k in m.PLAIN)
    assert pair_count == {k: 2 for k in m.PLAIN}
    first_four = Counter(s["label"] for t in ("T0_H214", "T1_H214", "T2_H214", "T3_H214") for s in by[t])
    assert first_four == {k: 1 for k in m.FUNCS}


def test_prompts_and_seeds():
    for s in STIM:
        assert s["nweights"] == 14 * s["H"] + 1 and f"### Network ({s['nweights']} weights)" in s["prompt"]
        assert s["prompt"].endswith("ANSWER: <one letter from A to Q>")
        assert m.WARNING not in s["prompt"] and s["prompt_warned"].replace(" " + m.WARNING, "") == s["prompt"]
        if s["seed"] is not None: assert s["seed"] >= 300000          # outside Pilot 2, round 2 and Stage 1 ranges
    assert len({s["prompt"] for s in STIM}) == len(STIM) and len({s["id"] for s in STIM}) == len(STIM)


# ---------------- runner and analysis ----------------
def test_call_plan_counts_and_order():
    jobs = rr.plan(rr.load())
    steps = Counter((j["step"], j["run"]) for j in jobs)
    assert [n for (_, _), n in sorted(steps.items())] == [48, 32, 16, 16, 48, 24, 16, 16, 16, 16]
    assert all(j["warned"] == j["run"].startswith("R3") for j in jobs)
    assert all(j["max_tokens"] == 6000 for j in jobs if j["model"] == "sonnet")
    assert len(jobs) == 248


def test_stats_match_scipy():
    from scipy.stats import binomtest, fisher_exact
    for b, c in [(14, 2), (5, 5), (0, 3), (9, 0)]:
        ref = binomtest(b, b + c, 0.5, alternative="greater").pvalue
        assert abs(rr.mcnemar_one_sided(b, c) - ref) < 1e-12
    for x1, n1, x2, n2 in [(19, 24, 6, 32), (10, 24, 10, 32), (24, 24, 0, 16)]:
        ref = fisher_exact([[x1, n1 - x1], [x2, n2 - x2]], alternative="greater").pvalue
        assert abs(rr.fisher_one_sided(x1, n1, x2, n2) - ref) < 1e-12


def test_tango_ci():
    lo, hi = rr.tango_ci(0, 0, 40)                           # no discordant pairs: closed form +-2.706/(n+2.706)
    assert abs(hi - 2.706 / 42.706) < 2e-3 and abs(lo + 2.706 / 42.706) < 2e-3
    lo, hi = rr.tango_ci(10, 2, 40)
    assert lo < 0.2 < hi and lo > 0                          # contains the estimate, excludes 0
    lo2, hi2 = rr.tango_ci(20, 4, 80)
    assert hi2 - lo2 < hi - lo                               # narrower with more data


def test_sign_flip_test():
    assert rr.sign_flip_p([0.5] * 8) == 1 / 256              # all 8 clusters above 0: exact minimum
    assert rr.sign_flip_p([0.0] * 8) == 1.0


def test_parse_answer_letters():
    assert rr.parse_answer("blah\nANSWER: Q") == "Q" and rr.parse_answer("ANSWER: **M**") == "M"
    assert rr.parse_answer("ANSWER: A ... later ANSWER: P") == "P"
    assert rr.parse_answer("ANSWER: R") is None and rr.parse_answer("no answer") is None


def test_report_runs_on_synthetic_results(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0); out = []
    for j in rr.plan(rr.load()):
        s = j["s"]; ans = s["label"] if rng.random() < 0.6 else str(rng.choice(list("ABCDEFGHIJKLMNOPQ")))
        out.append(dict(run=j["run"], model=j["model"], id=s["id"], sample=0, set=s["set"], network=s["network"],
                        H=s["H"], nweights=s["nweights"], version=s["version"], func=s["func"], label=s["label"],
                        bias_cue_ok=s["bias_cue_ok"], answer=ans, correct=ans == s["label"], cost=0.01,
                        stop_reason="end_turn", output_tokens=1000, model_id="x", text="the output bias is negative"))
    p = tmp_path / "r.jsonl"; p.write_text("\n".join(json.dumps(r) for r in out), encoding="utf-8")
    monkeypatch.setattr(rr, "RESULTS", p)
    rr.report(); txt = capsys.readouterr().out
    for tag in ("D1 ", "D1 verdict", "D1-cue", "D1-native", "D2 ", "D2 verdict", "D3 ", "D3 verdict", "D4 Opus",
                "D5 R2_haiku_low_6k", "D6 ", "Plain-option share", "Replies that mention"):
        assert tag in txt, tag
