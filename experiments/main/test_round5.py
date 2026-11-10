"""Round 5 checks (no API calls). Run: python -m pytest test_round5.py -q"""
import json, random, time
from collections import Counter

import numpy as np

import make_round3 as m
import make_round5 as m5
import run_round3 as base
import run_round5 as r5

S5 = r5.load5()
net = lambda s: m.parse_block(s["prompt"].split("### Network", 1)[1])
OLD3 = {s["id"]: s for s in m5.jl("stimuli_round3.jsonl")}
OLD4 = {s["id"]: s for s in m5.jl("stimuli_round4.jsonl")}


def test_counts():
    assert Counter(s["set"] for s in S5) == {"MONO-CUE-214": 24, "NEG-214": 36, "TRUNKR-214-NOQ": 64}
    assert len({s["id"] for s in S5}) == len(S5)


def test_mono_cue_labels_correct_on_rounded_weights_and_cue_reversed():
    mc = [s for s in S5 if s["set"] == "MONO-CUE-214"]
    assert Counter((s["func"], s["version"]) for s in mc) == {(k, v): 2 for k in "ABDIJK" for v in ("orig", "neg")}
    by = {}
    for s in mc:
        n = net(s)
        assert m.label(n) == s["label"] == (s["func"] if s["version"] == "orig" else m.COMPLEMENT[s["func"]]), s["id"]
        assert s["bias_cue_ok"] is False and m.bias_cue_ok(n, s["label"]) is False, s["id"]
        assert m.REVERSED(n if s["version"] == "orig" else m.negate(n)), s["id"]     # the CUE-214 definition
        by.setdefault(s["network"], {})[s["version"]] = n
    assert len(by) == 12
    for nm, v in by.items():          # the negated version is the exact negation of the original as displayed
        o, g = v["orig"], v["neg"]
        assert np.array_equal(o[0], g[0]) and np.array_equal(o[1], g[1]) and np.array_equal(o[2], -g[2]) and o[3] == -g[3], nm


def test_mono_cue_seeds_and_prompts_are_new():
    old_seeds = {s["seed"] for s in list(OLD3.values()) + list(OLD4.values()) if s["seed"] is not None}
    for s in S5:
        if s["set"] == "MONO-CUE-214": assert s["seed"] >= 900000 and s["seed"] not in old_seeds, s["id"]
    mc = {s["prompt"] for s in S5 if s["set"] == "MONO-CUE-214"}
    assert len(mc) == 24 and not mc & {s["prompt"] for s in list(OLD3.values()) + list(OLD4.values())}


def test_neg_mono_rows_are_round3_rows_unchanged():
    neg = [s for s in S5 if s["set"] == "NEG-214"]
    assert all(s == OLD3[s["id"]] for s in neg)
    assert Counter(s["func"] for s in neg) == {k: 6 for k in "ABDIJK"}
    assert sum(s["bias_cue_ok"] for s in neg) == 34


def test_noq_prompts_have_no_q_and_ask_for_a_to_p():
    nq = [s for s in S5 if s["set"] == "TRUNKR-214-NOQ"]
    assert len({s["network"] for s in nq}) == 16
    for s in nq:
        src = OLD4[s["src"]]
        for p, p0 in ((s["prompt"], src["prompt"]), (s["prompt_warned"], src["prompt_warned"])):
            assert "\nQ: " not in p and "including Q" not in p and "none of the above" not in p
            assert p.endswith("ANSWER: <one letter from A to P>") and "A to Q" not in p
            back = (p.replace("ANSWER: <one letter from A to P>", "ANSWER: <one letter from A to Q>")
                     .replace("Any option may be correct.", "Any option may be correct, including Q.")
                     .replace("(1 if an even number are 1)\n", "(1 if an even number are 1)\nQ: " + m.OPTIONS["Q"][0] + "\n", 1))
            assert back == p0, s["id"]          # nothing else changes
        assert {k: v for k, v in s.items() if k not in ("id", "set", "src", "prompt", "prompt_warned")} == \
               {k: v for k, v in src.items() if k not in ("id", "set", "src", "prompt", "prompt_warned")}
        assert m.label(net(s)) == s["label"] != "Q"


def test_mono_verdict_on_toy_inputs():
    assert r5.mono_verdict(12, 24, 46, 46, 20)[0] == "MONOTONE POLARITY FOLLOWS THE BIAS SIGN"
    assert r5.mono_verdict(21, 24, 46, 46, 3)[0].startswith("COMPUTES MONOTONE POLARITY")       # 87.5% >= 85%
    assert r5.mono_verdict(12, 24, 46, 46, 10)[0].startswith("INCONCLUSIVE: the new networks are harder")
    assert r5.mono_verdict(19, 24, 46, 46, 24)[0] == "INCONCLUSIVE"                                # drop 21 points
    assert r5.mono_verdict(16, 24, 22, 24, 24)[0] == "INCONCLUSIVE"     # drop 25 points < 30
    v, drop, p = r5.mono_verdict(12, 24, 46, 46, 20)
    assert abs(drop - 0.5) < 1e-12 and abs(p - base.fisher_one_sided(46, 46, 12, 24)) < 1e-15 and p < 1e-4
    # drop >= 30 points but not significant: 2/3 vs 3/3
    assert r5.mono_verdict(2, 3, 3, 3, 3)[0] == "INCONCLUSIVE"


def test_trunk_verdict_and_net_perm_on_toy_inputs():
    assert r5.trunk_verdict(0.01, 0.6) == "OPUS DECODES READOUTS UNDER FORCED CHOICE"
    assert r5.trunk_verdict(0.20, 0.39) == "NO DECODING EVEN UNDER FORCED CHOICE"
    assert r5.trunk_verdict(0.20, 0.44) == "INCONCLUSIVE"
    row = lambda nw, ok: dict(network=nw, answer="A" if ok else "E", label="A")
    hi = [row(f"h{i}", True) for i in range(3) for _ in range(2)]
    lo = [row(f"l{i}", False) for i in range(3) for _ in range(2)]
    p, na, nb = r5.net_perm(hi, lo, reps=20000)
    assert (na, nb) == (3, 3) and abs(p - 1 / 20) < 0.01          # perfect separation, 3 vs 3: exact p = 1/C(6,3)
    assert r5.net_perm(hi, hi, reps=2000)[0] == 1.0
    assert r5.pair_ok(dict(answer="E", label="A")) and not r5.pair_ok(dict(answer="B", label="A"))
    assert not r5.pair_ok(dict(answer=None, label="A")) and not r5.pair_ok(dict(answer="Q", label="A"))


def test_plan_counts(monkeypatch):
    monkeypatch.setattr(base, "STEPS", r5.STEPS_5)
    assert [n for _, n in sorted(Counter(j["step"] for j in base.plan(S5)).items())] == [24, 24, 36, 64]


def _fake_call(client, job):
    time.sleep(random.random() * 0.01)
    s = job["s"]
    return dict(answer=s["label"], stop_reason="end_turn", model_id="fake", input_tokens=17000, output_tokens=3000,
                seconds=0.0, text="")


def test_budget_guard_stops_at_cap(tmp_path, monkeypatch, capsys):
    cap = 5.00
    for k, v in dict(STEPS=r5.STEPS_5, RESULTS=tmp_path / "r.jsonl", BUDGET_USD=cap, load=r5.load5,
                     report=lambda: None, Budget=r5.FifoBudget, call=_fake_call, make_client=lambda: None).items():
        monkeypatch.setattr(base, k, v)
    base.run(type("A", (), {"dry_run": False})())
    done = [json.loads(l) for l in (tmp_path / "r.jsonl").read_text(encoding="utf-8").splitlines()]
    jobs = base.plan(S5); keys = {(r["run"], r["id"]) for r in done}
    skipped = [j for j in jobs if (j["run"], j["s"]["id"]) not in keys]
    spent = sum(r["cost"] for r in done)
    worst = lambda j: base.cost(j["model"], base.est_tokens(j["s"]["prompt_warned"]), j["max_tokens"])
    assert spent <= cap and skipped                                   # stayed under the cap, and the cap bound
    assert spent + worst(skipped[0]) > cap - 1e-3                     # stopped only when the next call could not fit
    first_cut = min(j["step"] for j in skipped)
    assert all(r["step"] <= first_cut for r in done)                  # nothing after the first cut ran
    assert all(j["step"] >= first_cut for j in skipped)               # earlier steps finished in full
    assert "calls skipped to stay under" in capsys.readouterr().out


def test_report_on_synthetic(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(base, "STEPS", r5.STEPS_5)
    rng = np.random.default_rng(0); out = []
    for j in base.plan(S5):
        s = j["s"]; a = s["label"] if rng.random() < 0.5 else str(rng.choice(list("ABCDEFGHIJKLMNOP")))
        out.append(dict(run=j["run"], model=j["model"], id=s["id"], sample=0, set=s["set"], network=s["network"],
                        version=s["version"], func=s["func"], label=s["label"], bias_cue_ok=s["bias_cue_ok"], answer=a,
                        correct=a == s["label"], cost=0.01, stop_reason="end_turn", output_tokens=1000, model_id="x"))
    p = tmp_path / "r.jsonl"; p.write_text("\n".join(json.dumps(r) for r in out), encoding="utf-8")
    monkeypatch.setattr(r5, "RESULTS_5", p)
    r5.report5(); txt = capsys.readouterr().out
    for tag in ("G1 Sonnet medium:", "G1 Sonnet medium verdict", "G2 Opus low:", "G2 Opus low verdict",
                "Opus on all 36 NEG-214", "G3 Opus low, forced choice (no Q) (16 trunks)", "G3 verdict",
                "F4a Opus low, with Q (16 trunks)", "on the 15 readouts answered Q in round 4"):
        assert tag in txt, tag
