#!/usr/bin/env python3
"""Every number and figure in the paper, computed from the saved API replies and stimuli (no API calls).

  python analysis.py          # writes numbers.tex (LaTeX macros), numbers.json and figures/*.pdf|png

Data: ../experiments/pilot (pilot; flip_followup/ for stimuli only), ../experiments/main (rounds 3-5)."""
import json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R3 = ROOT / "experiments" / "main"
PILOT_DIR = ROOT / "experiments" / "pilot"
sys.path.insert(0, str(R3))
import make_round3 as m          # noqa: E402  (options, parser, forward pass)
import run_round3 as base        # noqa: E402  (tests and helpers)
import run_round4 as r4          # noqa: E402

PLAIN, GROUP, PAIR = base.PLAIN, base.GROUP, base.PAIR
XL_FUNCS = set("CGLP")           # XOR, XNOR, parity, even parity
N = {}                           # macro name -> text


def jl(p): return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def frac(x, n): return f"{x}/{n}"


def pctf(x, n): return f"{100 * x / n:.0f}\\%"


def sci(p):
    mant, exp = f"{p:.1e}".split("e")
    return r"\ensuremath{" + mant + r" \times 10^{" + str(int(exp)) + "}}"   # safe inside or outside $...$


def net_of(prompt): return m.parse_block(prompt.split("### Network", 1)[1])


# ---------------- data ----------------
RES = [r for f in ("results_round3.jsonl", "results_round3b.jsonl", "results_round3c.jsonl", "results_round4.jsonl")
       for r in jl(R3 / f)]
STIM = {s["id"]: s for f in ("stimuli_round3.jsonl", "stimuli_round3b.jsonl", "stimuli_round4.jsonl") for s in jl(R3 / f)}
PILOT = jl(PILOT_DIR / "results.jsonl")
SONNET = {"R1_sonnet_medium_6k", "R5_sonnet_medium_6k"}
OPUS_LOW = {"R4b_opus_low_8k", "R4c_opus_low_6k", "R5_opus_low_6k"}
SINGLE_214 = {"NEG-214", "CUE-214", "CUE2-214", "XL2-214", "NATIVE-214"}
pol = base.pol_ok
trunc = base.trunc


def runs(names, flt=lambda r: True): return [r for r in RES if r["run"] in names and flt(r)]


def cluster_ci(rows, key, stat=pol, reps=10000, seed=0):
    """90% percentile bootstrap over clusters (networks or trunks) of the pooled proportion (90%, as in the text)."""
    by = defaultdict(list)
    for r in rows: by[r[key]].append(stat(r))
    cl = list(by.values()); rng = random.Random(seed); out = []
    for _ in range(reps):
        s = [rng.choice(cl) for _ in cl]; out.append(sum(map(sum, s)) / sum(map(len, s)))
    out.sort()
    return out[int(0.05 * reps)], out[int(0.95 * reps) - 1]


# ---------------- 1. the pilot and its dead networks ----------------
def dead_stats(nets):
    ones = np.array([m.forward(n).mean() for n in nets])
    return int(np.sum((ones <= 0.02) | (ones >= 0.98))), int(np.sum((ones == 0) | (ones == 1))), len(nets)


def pilot_flip_nets():
    """Pilot broken (partial-flip) networks, keyed by the pilot's stimulus id."""
    P2, X = PILOT_DIR / "networks", PILOT_DIR / "extra"
    nets = {}
    key = json.loads((P2 / "answer_key.json").read_text())
    for st in (0, 1):
        for sid, bl in m._blocks(P2 / f"prompt_H214_set{st}.txt").items():
            if key[f"H214_set{st}_{sid}"]["version"] == "flip": nets[f"H214_set{st}_{sid}"] = m.parse_block(bl)
    xk = json.loads((X / "answer_key.json").read_text())
    for sid, bl in m._blocks(X / "prompt_extra.txt").items():
        if xk[f"X_{sid}"]["version"] == "flip": nets[f"X_{sid}"] = m.parse_block(bl)
    return nets


PFN = pilot_flip_nets()
near, exact, n = dead_stats(list(PFN.values()))
N.update(PilotDeadNear=frac(near, n), PilotDeadExact=str(exact))
s1 = [s for s in jl(PILOT_DIR / "flip_followup" / "stimuli_stage1.jsonl") if s["version"] == "flip"]
near1, _, n1 = dead_stats([net_of(s["prompt"]) for s in s1])
N.update(StageDeadNear=frac(near1, n1))
pm = [r for r in PILOT if r["run"] in ("sonnet_medium_6k", "sonnet_medium_6k_extra") and r["nweights"] == 2997]
pn = defaultdict(dict)
for r in pm: pn[r.get("network") or f"{r['func']}_H{r['H']}"][r["version"]] = r
pn = {k: v for k, v in pn.items() if len(v) == 2}
po, pf = sum(v["orig"]["correct"] for v in pn.values()), sum(v["flip"]["correct"] for v in pn.values())
b = sum(v["orig"]["correct"] and not v["flip"]["correct"] for v in pn.values()); c = sum((not v["orig"]["correct"]) and v["flip"]["correct"] for v in pn.values())
N.update(PilotOrig=frac(po, len(pn)), PilotCaught=frac(pf, len(pn)), PilotP=f"{base.mcnemar_one_sided(b, c):.3f}",
         PilotNamedOld=str(sum(v["flip"]["named_original"] for v in pn.values())))
# was the miss caused by the dead networks? caught / named-original rates on constant vs non-constant broken networks
ones_ = {i: m.forward(x).mean() for i, x in PFN.items()}
fl_ = [v["flip"] for v in pn.values()]
for tag, rr in (("Dead", [r for r in fl_ if ones_[r["id"]] <= 0.02 or ones_[r["id"]] >= 0.98]),
                ("Live", [r for r in fl_ if 0.02 < ones_[r["id"]] < 0.98])):
    N.update({f"Pilot{tag}Caught": frac(sum(r["correct"] for r in rr), len(rr)),
              f"Pilot{tag}Named": frac(sum(r["named_original"] for r in rr), len(rr))})
N["PilotOptions"] = str(len(m.OLD_TO_NEW) + 1)    # the pilot's list: 8 plain functions and "none"
# what the edit keeps: the output at x = 0, i.e. polarity (every pilot original is a plain function, so it is 0 before the edit)
N["PilotZeroSame"] = frac(sum(int(m.forward(x)[0]) == 0 for x in PFN.values()), len(PFN))

# ---------------- 2. D1: polarity on a clean break (Sonnet, NEG-214) ----------------
d = {r["id"]: r for r in runs({"R1_sonnet_medium_6k"})}
pairs = base.pairs_of(d, "NEG-214")
k = len(pairs); oc = sum(o["correct"] for o, _ in pairs); nc = sum(g["correct"] for _, g in pairs)
bb = sum(base.is_plain(o["answer"]) and not base.is_plain(g["answer"]) for o, g in pairs)
cc = sum(not base.is_plain(o["answer"]) and base.is_plain(g["answer"]) for o, g in pairs)
lo, hi = base.tango_ci(bb, cc, k)
N.update(DOneOrig=frac(oc, k), DOneNeg=frac(nc, k), DOneS=f"{100 * (bb - cc) / k:+.0f}", DOneLo=f"{100 * lo:+.0f}",
         DOneHi=f"{100 * hi:+.0f}", DOneB=str(bb), DOneC=str(cc), DOneP=f"{base.mcnemar_one_sided(bb, cc):.4f}",
         DOneNamed=frac(sum(g["answer"] == g["func"] for _, g in pairs), k),
         DOneBoth=frac(sum(o["correct"] and g["correct"] for o, g in pairs), k))

# ---------------- 3. gate family and model (single-task 214-unit networks, original + negated) ----------------
fam = lambda r: "xl" if r["func"] in XL_FUNCS else "mono"
son = runs(SONNET, lambda r: r["set"] in SINGLE_214 and r["version"] in ("orig", "neg"))
opl = runs(OPUS_LOW, lambda r: r["set"] in SINGLE_214 and r["version"] in ("orig", "neg"))
for tag, rows in (("SonMono", [r for r in son if fam(r) == "mono"]), ("SonXL", [r for r in son if fam(r) == "xl"]),
                  ("OpusXL", [r for r in opl if fam(r) == "xl"])):
    N[tag + "Pol"] = frac(sum(map(pol, rows)), len(rows)); N[tag + "PolPct"] = pctf(sum(map(pol, rows)), len(rows))
    N[tag + "Exact"] = frac(sum(r["correct"] for r in rows), len(rows)); N[tag + "ExactPct"] = pctf(sum(r["correct"] for r in rows), len(rows))
    N[tag + "Nets"] = str(len({r["network"] for r in rows}))
# gate family vs cue validity (Sonnet): polarity when the bias sign points the right way
for tag, fm in (("Mono", "mono"), ("XL", "xl")):
    rr = [r for r in son if fam(r) == fm and r["bias_cue_ok"]]
    N[f"Son{tag}CueOkPol"] = frac(sum(map(pol, rr)), len(rr))
sxl = [r for r in son if fam(r) == "xl"]
N["SonXLAnsPlain"] = frac(sum(base.is_plain(r["answer"]) for r in sxl), len(sxl))
N["SonXLLabPlain"] = frac(sum(r["label"] in PLAIN for r in sxl), len(sxl))
# F3 (confirmatory, round 4) and E3 (round 3b)
xs = runs({"R5_sonnet_medium_6k"}, lambda r: r["set"] == "XL2-214"); xo = runs({"R5_opus_low_6k"}, lambda r: r["set"] == "XL2-214")
ns_, no_ = defaultdict(int), defaultdict(int)
for r in xs: ns_[r["network"]] += pol(r)
for r in xo: no_[r["network"]] += pol(r)
nets = sorted(set(ns_) & set(no_)); dd = [no_[x] - ns_[x] for x in nets]
N.update(FThreeOpus=frac(sum(no_[x] for x in nets), 2 * len(nets)), FThreeSon=frac(sum(ns_[x] for x in nets), 2 * len(nets)),
         FThreeGap=f"{100 * sum(dd) / (2 * len(nets)):+.0f}", FThreeP=sci(r4.sign_flip_exact(dd, 1)),
         FThreeNets=str(len(nets)), FThreeOpusExact=frac(sum(r["correct"] for r in xo), len(xo)),
         FThreeSonExact=frac(sum(r["correct"] for r in xs), len(xs)))
e3 = runs({"R4b_opus_low_8k"}); e3s = [d[r["id"]] for r in e3 if r["id"] in d]
N.update(EThreeOpus=frac(sum(map(pol, e3)), len(e3)), EThreeSon=frac(sum(map(pol, e3s)), len(e3s)),
         EThreeOpusExact=frac(sum(r["correct"] for r in e3), len(e3)))

# bias cue (exploratory pooled; excludes NATIVE, whose cue is almost always right)
cue_sets = {"NEG-214", "CUE-214", "CUE2-214", "XL2-214"}
def cue_split(rows):
    rows = [r for r in rows if r["set"] in cue_sets and fam(r) == "xl"]
    rt, wr = [r for r in rows if r["bias_cue_ok"]], [r for r in rows if not r["bias_cue_ok"]]
    return rt, wr
def net_perm(rt, wr, reps=200000, seed=0):
    a = defaultdict(int); bnet = defaultdict(int)
    for r in rt: a[r["network"]] += pol(r)
    for r in wr: bnet[r["network"]] += pol(r)
    av, bv = list(a.values()), list(bnet.values()); obs = sum(av) / len(av) - sum(bv) / len(bv)
    allv = av + bv; rng = random.Random(seed); hits = 0
    for _ in range(reps):
        rng.shuffle(allv); hits += sum(allv[:len(av)]) / len(av) - sum(allv[len(av):]) / len(bv) >= obs - 1e-12
    return hits / reps, len(av), len(bv)
for tag, rows in (("Son", son), ("Opus", opl)):
    rt, wr = cue_split(rows); p, na, nb = net_perm(rt, wr)
    N.update({f"{tag}CueRight": frac(sum(map(pol, rt)), len(rt)), f"{tag}CueWrong": frac(sum(map(pol, wr)), len(wr)),
              f"{tag}CueP": f"{p:.3f}", f"{tag}CueNets": f"{na} vs {nb}"})
# Sonnet's split on round 4's new networks (held out from the rounds that suggested the shortcut) vs rounds 3/3b
for tag, keep in (("Four", lambda r: r["set"] == "XL2-214"), ("Early", lambda r: r["set"] != "XL2-214")):
    rt, wr = cue_split([r for r in son if keep(r)]); p, na, nb = net_perm(rt, wr)
    N.update({f"SonCue{tag}Right": frac(sum(map(pol, rt)), len(rt)), f"SonCue{tag}Wrong": frac(sum(map(pol, wr)), len(wr)),
              f"SonCue{tag}P": f"{p:.2f}"})
# share of Sonnet's XOR/parity answers that follow the cue, from a follow-cue / follow-truth / guess mixture:
# P(right | cue right) - P(right | cue wrong) = share following the cue
rt, wr = cue_split(son)
pr_, pw_ = sum(map(pol, rt)) / len(rt), sum(map(pol, wr)) / len(wr)
N["SonCueFollowPct"] = f"{100 * (pr_ - pw_):.0f}\\%"
N["SonCueTruthPct"] = f"{max(0, 100 * (2 * pw_ - 1 + pr_ - pw_)):.0f}\\%"   # P(wrong-cue right) = truth + guess/2

# ---------------- 4. shared trunks ----------------
def trunk_stats(rows):
    per = defaultdict(list)
    for r in rows: per[r["network"]].append(r)
    full = [v for _, v in sorted(per.items()) if len(v) == 4]; t = [r for v in full for r in v]   # sorted: same bootstrap as run_round4
    accs = [sum(r["correct"] for r in v) / 4 for v in full]
    lo_, hi_ = r4.boot_ci(accs)
    grp = [r for r in t if r["answer"] in GROUP and GROUP[r["answer"]] == GROUP[r["label"]]]
    pr = [r for r in grp if PAIR[r["answer"]] == PAIR[r["label"]]]
    in_grp = lambda r: r["answer"] in GROUP and GROUP[r["answer"]] == GROUP[r["label"]]
    gacc = [sum(map(in_grp, v)) for v in full]
    exp_pair = 0.0     # pair-right count expected if, within each group, answers were independent of the readout
    for gg in set(GROUP.values()):
        gr = [r for r in grp if GROUP[r["label"]] == gg]
        if gr:
            ap, lp = Counter(PAIR[r["answer"]] for r in gr), Counter(PAIR[r["label"]] for r in gr)
            exp_pair += sum(ap[x] * lp[x] for x in ap) / len(gr)
    return dict(n=len(full), exact=sum(r["correct"] for r in t), N=len(t), mean=sum(accs) / len(accs), lo=lo_, hi=hi_,
                p=r4.sign_flip_exact([a - 0.25 for a in accs], 4), grp=len(grp), pair=len(pr),
                polpair=sum(r["correct"] for r in pr), rows=t,
                p_eighth=r4.sign_flip_exact([a - 0.125 for a in accs], 8),       # exploratory: polarity-only baseline
                p_grp=r4.sign_flip_exact([g - 2 for g in gacc], 1), exp_pair=exp_pair,
                distinct3=sum(len({r["answer"] for r in v}) >= 3 for v in full),
                trunc=sum(map(trunc, t)), none=sum(r["answer"] == "Q" for r in t),
                pol_all=sum(map(pol, t)))
TR = {"Son": trunk_stats(runs({"R5_sonnet_medium_6k"}, lambda r: r["set"] == "TRUNKR-214")),
      "Opus": trunk_stats(runs({"R5_opus_low_6k"}, lambda r: r["set"] == "TRUNKR-214"))}
for tag, t in TR.items():
    N.update({f"{tag}TrExact": frac(t["exact"], t["N"]), f"{tag}TrMean": f"{100 * t['mean']:.0f}\\%",
              f"{tag}TrLo": f"{100 * t['lo']:.0f}", f"{tag}TrHi": f"{100 * t['hi']:.0f}", f"{tag}TrP": f"{t['p']:.2f}",
              f"{tag}TrGrp": frac(t["grp"], t["N"]), f"{tag}TrPair": frac(t["pair"], t["grp"]),
              f"{tag}TrPolPair": frac(t["polpair"], t["pair"]), f"{tag}TrTrunks": str(t["n"]),
              f"{tag}TrPEighth": f"{t['p_eighth']:.3f}", f"{tag}TrGrpP": f"{t['p_grp']:.3f}",
              f"{tag}TrPairExp": f"{t['exp_pair']:.0f}", f"{tag}TrDistinct": frac(t["distinct3"], t["n"]),
              f"{tag}TrTrunc": frac(t["trunc"], t["N"]), f"{tag}TrNoneR": frac(t["none"], t["N"]),
              f"{tag}TrPolAll": frac(t["pol_all"], t["N"])})
st_ = TR["Son"]["rows"]; named_ = [r for r in st_ if r["answer"] in GROUP]
N["SonTrThreeGroup"] = frac(sum(GROUP[r["answer"]] == GROUP["I"] for r in named_), len(named_))
N["SonTrCueRight"] = frac(sum(pol(r) for r in st_ if r["bias_cue_ok"]), sum(r["bias_cue_ok"] for r in st_))
N["SonTrCueWrong"] = frac(sum(pol(r) for r in st_ if not r["bias_cue_ok"]), sum(not r["bias_cue_ok"] for r in st_))
old_s = trunk_stats(runs({"R1_sonnet_medium_6k"}, lambda r: r["set"] == "TRUNK-214"))
old_o = trunk_stats(runs({"R4c_opus_low_6k"}, lambda r: r["set"] == "TRUNK-214"))
src = {s["id"]: s["src"] for s in STIM.values() if s.get("src") and s["set"] == "TRUNKR-214"}
for tag, r_new, r_old in (("Son", "R5_sonnet_medium_6k", "R1_sonnet_medium_6k"), ("Opus", "R5_opus_low_6k", "R4c_opus_low_6k")):
    old = {r["id"]: r for r in runs({r_old}, lambda r: r["set"] == "TRUNK-214")}
    pr_ = [(old[src[r["id"]]], r) for r in runs({r_new}, lambda r: r["set"] == "TRUNKR-214") if src.get(r["id"]) in old]
    N[f"{tag}TrScale"] = f"{sum(o['correct'] for o, _ in pr_)}/{len(pr_)} vs {sum(x['correct'] for _, x in pr_)}/{len(pr_)}"
N.update(SonTrOldExact=frac(old_s["exact"], old_s["N"]), OpusTrOldExact=frac(old_o["exact"], old_o["N"]),
         OpusTrOldNone=frac(old_o["none"], old_o["N"]))

# ---------------- 5. size, effort, other models, warning ----------------
d36 = base.pairs_of(d, "NEG-36")
N["SmallAll"] = frac(sum(o["correct"] + g["correct"] for o, g in d36), 2 * len(d36))
for run_, tag in (("R2_haiku_low_6k", "HaikuLow"), ("R2_haiku_medium_6k", "HaikuMed"), ("R4_opus_medium_8k", "OpusMed"),
                  ("R5_sonnet_high_12k", "SonHigh")):
    rr = runs({run_}); N[tag + "Trunc"] = frac(sum(map(trunc, rr)), len(rr))
N["SonHighMeanOut"] = f"{np.mean([r['output_tokens'] for r in runs({'R5_sonnet_high_12k'})]):,.0f}"
done_ = [r for r in runs({"R5_sonnet_high_12k"}) if not trunc(r)]
N["SonHighDoneRight"] = frac(sum(map(pol, done_)), len(done_))
N["OpusLowTrunc"] = frac(sum(map(trunc, runs(OPUS_LOW))), len(runs(OPUS_LOW)))
nat = base.pairs_of(d, "NATIVE-214")
N["NativePlain"] = frac(sum(base.is_plain(o["answer"]) for o, _ in nat), len(nat))
w = {r["id"]: r for r in runs({"R3_sonnet_medium_6k_warned"})}
wn = [(d[i], w[i]) for i in w if i in d and w[i]["set"] == "NEG-214" and w[i]["version"] == "neg"]
bw = sum(y["correct"] and not x["correct"] for x, y in wn); cw = sum(x["correct"] and not y["correct"] for x, y in wn)
lo, hi = base.tango_ci(bw, cw, len(wn))
N.update(WarnGain=f"{100 * (bw - cw) / len(wn):+.0f}", WarnLo=f"{100 * lo:+.1f}", WarnHi=f"{100 * hi:+.0f}")
N.update(WarnUnw=frac(sum(x["correct"] for x, _ in wn), len(wn)), WarnW=frac(sum(y["correct"] for _, y in wn), len(wn)))
# D1-cue (round 3) and its replication E1 (round 3b), Sonnet
r3cue = [r for r in runs({"R1_sonnet_medium_6k"}) if r["set"] in ("NEG-214", "CUE-214") and fam(r) == "xl" and r["version"] in ("orig", "neg")]
e1 = [r for r in runs({"R1_sonnet_medium_6k"}) if r["set"] == "CUE2-214"]
for tag, rows in (("DOneCue", r3cue), ("EOne", e1)):
    rt, wr = [r for r in rows if r["bias_cue_ok"]], [r for r in rows if not r["bias_cue_ok"]]
    p, _, _ = net_perm(rt, wr)
    N.update({f"{tag}Right": frac(sum(map(pol, rt)), len(rt)), f"{tag}Wrong": frac(sum(map(pol, wr)), len(wr)), f"{tag}P": f"{p:.2f}",
              f"{tag}Fisher": f"{base.fisher_one_sided(sum(map(pol, rt)), len(rt), sum(map(pol, wr)), len(wr)):.3f}"})
# E2: warning on cue-wrong XOR/parity stimuli
e2 = [(d[i], w[i]) for i in w if i in d and w[i]["func"] in XL_FUNCS and w[i]["version"] in ("orig", "neg") and not w[i]["bias_cue_ok"]
      and w[i]["set"] in ("NEG-214", "CUE-214", "CUE2-214")]
b2_ = sum(pol(y) and not pol(x) for x, y in e2); c2_ = sum(pol(x) and not pol(y) for x, y in e2); lo2, hi2 = base.tango_ci(b2_, c2_, len(e2))
N.update(ETwoUnw=frac(sum(pol(x) for x, _ in e2), len(e2)), ETwoW=frac(sum(pol(y) for _, y in e2), len(e2)),
         ETwoGain=f"{100 * (b2_ - c2_) / len(e2):+.0f}", ETwoLo=f"{100 * lo2:+.0f}", ETwoHi=f"{100 * hi2:+.0f}")
single_orig = [r for r in RES if r["version"] == "orig" and r["set"] in SINGLE_214 | {"NEG-36"}]
N["NoneFASingle"] = frac(sum(r["answer"] == "Q" for r in single_orig), len(single_orig))
for tag, names in (("Son", SONNET), ("Opus", OPUS_LOW)):
    tq = runs(names, lambda r: r["version"] == "trunk"); so_ = runs(names, lambda r: r["version"] == "orig" and r["set"] in SINGLE_214)
    N[tag + "TrNone"] = frac(sum(r["answer"] == "Q" for r in tq), len(tq)); N[tag + "SingleNone"] = frac(sum(r["answer"] == "Q" for r in so_), len(so_))
    nq = [r for r in TR[tag]["rows"] if r["answer"] != "Q"]
    N[tag + "TrExactAnswered"] = frac(sum(r["correct"] for r in nq), len(nq))

# ---------------- 6. costs ----------------
allr = RES + PILOT
N["TotalCalls"] = f"{len(allr):,}"; N["TotalCost"] = f"\\${sum(r['cost'] for r in allr):.2f}"
N["HaikuCalls"] = str(sum(r.get("model") == "haiku" for r in allr)); N["MainCalls"] = str(sum(r.get("model") != "haiku" for r in allr))
d36_cue = [s for s in STIM.values() if s["set"] == "NEG-36"]
N["SmallCueOk"] = frac(sum(s["bias_cue_ok"] for s in d36_cue), len(d36_cue))
for tag, nm, flt in (("Son", "sonnet", lambda r: r["run"] in SONNET and r["nweights"] == 2997),
                     ("Opus", "opus", lambda r: r["run"] in OPUS_LOW and r["nweights"] == 2997)):
    rr = [r for r in RES if flt(r)]
    N[tag + "CallCost"] = f"\\${np.mean([r['cost'] for r in rr]):.3f}"; N[tag + "MeanOut"] = f"{np.mean([r['output_tokens'] for r in rr]):,.0f}"
N["MeanIn"] = f"{np.mean([r['input_tokens'] for r in RES if r['nweights'] == 2997]):,.0f}"

# ---------------- 7. offline baselines (no model) ----------------
def group_by_norm(net):
    W1 = net[0]; a = np.abs(W1[[2, 5]]).mean(); b_ = np.abs(W1[[0, 3, 7]]).mean()
    return "x2,x5" if a > b_ else "x0,x3,x7"
single = [s for s in STIM.values() if s["set"] in SINGLE_214]
trunks = [s for s in STIM.values() if s["set"] == "TRUNKR-214"]
N["BaseNormGroupSingle"] = pctf(sum(group_by_norm(net_of(s["prompt"])) == GROUP[s["label"]] for s in single), len(single))
N["BaseNormGroupTrunk"] = pctf(sum(group_by_norm(net_of(s["prompt"])) == GROUP[s["label"]] for s in trunks), len(trunks))
xl_cue = [s for s in single if s["set"] in cue_sets and s["func"] in XL_FUNCS]
N["BaseBiasXL"] = pctf(sum(s["bias_cue_ok"] for s in xl_cue), len(xl_cue))
mono = [s for s in single if s["func"] not in XL_FUNCS]
N["BaseBiasMono"] = pctf(sum(s["bias_cue_ok"] for s in mono), len(mono))
zero_ok = all((m.forward(net_of(s["prompt"]))[0] == 1) == (s["label"] not in PLAIN) for s in list(STIM.values()))
assert zero_ok          # polarity is always decidable from the all-zero input
N["NStimuli"] = str(len(STIM))
# polarity = output at x = 0 = sign(b2 + sum_j W2_j relu(b1_j)). How many terms does a reader need?
# Rule: b2 plus the k hidden units with the largest |contribution| at x = 0 (k = 0 is the bias sign alone).
xl_all = [s for s in single if s["set"] in cue_sets and s["func"] in XL_FUNCS]
for k_, nm in ((0, "Zero"), (1, "One"), (5, "Five"), (10, "Ten")):
    ok = 0
    for s in xl_all:
        W1_, b1_, W2_, b2_ = net_of(s["prompt"]); con = W2_ * np.maximum(0, b1_)
        top = np.argsort(-np.abs(con))[:k_]
        ok += (b2_ + con[top].sum() > 0) == (s["label"] not in PLAIN)
    N[f"BaseTop{nm}"] = pctf(ok, len(xl_all))
N["BaseTopN"] = str(len(xl_all))
N["BaseActiveAtZero"] = f"{np.mean([(net_of(s['prompt'])[1] > 0).sum() for s in xl_all]):.0f}"
# direction cue: plain iff W2 . sum_i s_i W1[i] > 0, with s_i = +1 for the gate's inputs (-1 for x5 in x2 AND NOT x5)
PLAIN_OF = dict(zip("EFGHMNOP", "ABCDIJKL"))
DIR_SIGNS = {"A": {2: 1, 5: 1}, "B": {2: 1, 5: 1}, "C": {2: 1, 5: 1}, "D": {2: 1, 5: -1}}


def dir_ok(s):
    W1_, _, W2_, _ = net_of(s["prompt"]); sg = DIR_SIGNS.get(PLAIN_OF.get(s["label"], s["label"]), {0: 1, 3: 1, 7: 1})
    return (float(W2_ @ sum(v * W1_[i] for i, v in sg.items())) > 0) == (s["label"] in PLAIN)


N["BaseDirMono"] = frac(sum(map(dir_ok, mono)), len(mono))
N["BaseDirXL"] = frac(sum(map(dir_ok, xl_all)), len(xl_all))
# what the replies say (descriptive regex counts on XOR/parity single-task replies)
ZERO = re.compile(r"all[- ]zeros?\b|x\s*=\s*0\b|\(0,\s*0\)|all inputs (are )?(0|zero)|zero input|input 0+\b|0000", re.I)
BIAS = re.compile(r"output bias|bias of the output|out(put)?[^.\n]{0,25}bias", re.I)
for tag, rows in (("Son", son), ("Opus", opl)):
    xr = [r for r in rows if fam(r) == "xl" and r.get("text")]
    N[tag + "XLBiasMention"] = frac(sum(bool(BIAS.search(r["text"])) for r in xr), len(xr))
    N[tag + "XLZeroMention"] = frac(sum(bool(ZERO.search(r["text"])) for r in xr), len(xr))
for tag, sets in (("Single", {"NEG-214"}), ("Trunk", {"TRUNK-214"}), ("TrunkR", {"TRUNKR-214"})):
    nets_ = [net_of(s["prompt"]) for s in STIM.values() if s["set"] in sets and s["version"] in ("orig", "trunk")]
    for nm, idx in (("Wone", 0), ("Bone", 1), ("Wtwo", 2)):
        N[f"Stat{tag}{nm}"] = f"{np.mean([np.abs(x[idx]).mean() for x in nets_]):.2f}"
    N[f"Stat{tag}Btwo"] = f"{np.mean([abs(x[3]) for x in nets_]):.2f}"

# ---------------- 8. round 5 (new macros only, all prefixed RFive; nothing above uses round-5 data) ----------------
def round5():
    import run_round5 as r5
    res5, stim5 = jl(R3 / "results_round5.jsonl"), jl(R3 / "stimuli_round5.jsonl")
    pfmt = lambda p: sci(p) if p < 0.001 else f"{p:.3f}"
    sel = lambda run, st: [r for r in res5 if r["run"] == run and r["set"] == st]
    mc = [s for s in stim5 if s["set"] == "MONO-CUE-214"]
    # construction (searched seeds and hits per method are stored with each MONO-CUE stimulus)
    srch = {s["func"]: s["search"] for s in mc}
    N["RFiveSearchN"] = f"{int(srch['D']['a'].split('/')[1]):,}"
    N["RFiveSearchNB"] = srch["A"]["b"].split("/")[1]
    N["RFiveHitsD"], N["RFiveHitsK"] = srch["D"]["a"].split("/")[0], srch["K"]["a"].split("/")[0]
    flips = [s["dead_flipped"] for s in mc if s["version"] == "orig" and "dead_flipped" in s]
    N["RFiveFlipsLo"], N["RFiveFlipsHi"] = str(min(flips)), str(max(flips))
    FN = dict(A="AND", B="OR", D="AND-NOT", I="MAJORITY", J="3-AND", K="3-OR")
    N["RFiveMethods"] = ", ".join(f"{FN[k]}~({srch_m})" for k, srch_m in sorted({(s['func'], s['method']) for s in mc}))
    N["RFiveDirCue"] = frac(sum(map(dir_ok, mc)), len(mc))
    always = lambda x: int((x[1] + np.minimum(x[0], 0).sum(0) > 0).sum())
    for tag, ss in (("Cue", [s for s in mc if s["version"] == "orig"]),
                    ("NegMono", [s for s in stim5 if s["set"] == "NEG-214" and s["version"] == "orig"])):
        nets_ = [net_of(s["prompt"]) for s in ss]
        for nm, f_ in (("Wone", lambda x: np.abs(x[0]).mean()), ("Bone", lambda x: np.abs(x[1]).mean()),
                       ("Wtwo", lambda x: np.abs(x[2]).mean()), ("Btwo", lambda x: abs(x[3])),
                       ("MaxBone", lambda x: np.abs(x[1]).max()), ("PosWtwo", lambda x: 100 * (x[2] > 0).mean())):
            N[f"RFiveStat{tag}{nm}"] = f"{np.mean([f_(x) for x in nets_]):.2f}" if nm != "PosWtwo" else f"{np.mean([f_(x) for x in nets_]):.0f}\\%"
        N[f"RFiveStat{tag}Always"] = f"{np.mean([always(x) for x in nets_]):.1f}"
    # G1 (Sonnet) and G2 (Opus): polarity on MONO-CUE vs the same model's cue-right monotone stimuli
    mono_ = lambda r: r["func"] not in XL_FUNCS and r["version"] in ("orig", "neg")
    son_base = runs({"R1_sonnet_medium_6k"}, lambda r: r["set"] in ("NEG-214", "NATIVE-214") and mono_(r) and r["bias_cue_ok"])
    opus_neg = sel("R6_opus_low_6k", "NEG-214")
    for tag, cue, basel in (("Son", sel("R6_sonnet_medium_6k", "MONO-CUE-214"), son_base),
                            ("Opus", sel("R6_opus_low_6k", "MONO-CUE-214"), [r for r in opus_neg if r["bias_cue_ok"]])):
        n_, nb = len(cue), len(basel); pc, pb = sum(map(pol, cue)), sum(map(pol, basel))
        pair_ = sum(map(r5.pair_ok, cue)); ex = sum(r["correct"] for r in cue)
        _, drop, p_ = r5.mono_verdict(pc, n_, pb, nb, pair_); pp, _, _ = r5.net_perm(basel, cue)
        N.update({f"RFive{tag}CuePol": frac(pc, n_), f"RFive{tag}CuePolPct": pctf(pc, n_), f"RFive{tag}BasePol": frac(pb, nb),
                  f"RFive{tag}Drop": f"{100 * drop:.0f}", f"RFive{tag}FisherP": pfmt(p_), f"RFive{tag}PermP": pfmt(pp),
                  f"RFive{tag}Pair": frac(pair_, n_), f"RFive{tag}PairPct": pctf(pair_, n_), f"RFive{tag}Exact": frac(ex, n_),
                  f"RFive{tag}ExactPct": pctf(ex, n_),
                  f"RFive{tag}Flipped": frac(sum(r5.pair_ok(r) and not r["correct"] for r in cue), n_ - ex),
                  f"RFive{tag}Trunc": frac(sum(map(trunc, cue)), n_)})
    # descriptive splits by construction method: (a)/(b) have no shifted units; (d) has flipped dead-unit weights
    meth = {s["id"]: s["method"] for s in mc}
    BIAS_ = re.compile(r"output bias|bias of the output|out(put)?[^.\n]{0,25}bias", re.I)
    for tag, run in (("Son", "R6_sonnet_medium_6k"), ("Opus", "R6_opus_low_6k")):
        cue = sel(run, "MONO-CUE-214")
        ns = [r for r in cue if meth[r["id"]] in "ab"]; dd = [r for r in cue if meth[r["id"]] == "d"]
        nd = [r for r in cue if meth[r["id"]] != "d"]
        N.update({f"RFive{tag}NoShiftPol": frac(sum(map(pol, ns)), len(ns)),
                  f"RFive{tag}FlipExact": frac(sum(r["correct"] for r in dd), len(dd)),
                  f"RFive{tag}OtherExact": frac(sum(r["correct"] for r in nd), len(nd)),
                  f"RFive{tag}BiasMention": frac(sum(bool(BIAS_.search(r["text"])) for r in cue), len(cue))})
    for tag, run in (("Son", "R6_sonnet_medium_6k"), ("Opus", "R6_opus_low_6k")):
        for vt, ver in (("Orig", "orig"), ("Neg", "neg")):
            x_ = [r for r in sel(run, "MONO-CUE-214") if r["version"] == ver]
            N[f"RFive{tag}{vt}Pol"] = frac(sum(map(pol, x_)), len(x_))
    new5 = [s for s in stim5 if s["id"] not in STIM]
    assert all((m.forward(net_of(s["prompt"]))[0] == 1) == (s["label"] not in PLAIN) for s in new5)
    N["RFiveNStimuli"] = str(len(STIM) + len(new5))
    cw = [r for r in opus_neg if not r["bias_cue_ok"]]
    N.update(RFiveOpusNegMonoPol=frac(sum(map(pol, opus_neg)), len(opus_neg)),
             RFiveOpusNegMonoExact=frac(sum(r["correct"] for r in opus_neg), len(opus_neg)),
             RFiveOpusNegCueWrongPol=frac(sum(map(pol, cw)), len(cw)))
    # G3: Opus on the rescaled trunks under forced choice
    nq = sel("R6_opus_low_6k", "TRUNKR-214-NOQ"); t5 = trunk_stats(nq)
    N.update(RFiveTrTrunks=str(t5["n"]), RFiveTrExact=frac(t5["exact"], t5["N"]), RFiveTrMean=f"{100 * t5['mean']:.0f}\\%",
             RFiveTrLo=f"{100 * t5['lo']:.0f}", RFiveTrHi=f"{100 * t5['hi']:.0f}", RFiveTrP=pfmt(t5["p"]),
             RFiveTrPEighth=pfmt(t5["p_eighth"]), RFiveTrGrp=frac(t5["grp"], t5["N"]), RFiveTrGrpP=pfmt(t5["p_grp"]),
             RFiveTrPair=frac(t5["pair"], t5["grp"]), RFiveTrPairExp=f"{t5['exp_pair']:.0f}",
             RFiveTrPolPair=frac(t5["polpair"], t5["pair"]), RFiveTrTrunc=frac(t5["trunc"], t5["N"]),
             RFiveTrQ=frac(t5["none"], t5["N"]), RFiveTrPolAll=frac(t5["pol_all"], t5["N"]),
             RFiveTrDistinct=frac(t5["distinct3"], t5["n"]))
    old = {r["id"]: r for r in runs({"R5_opus_low_6k"}, lambda r: r["set"] == "TRUNKR-214")}
    pr5 = [(old[r["id"][:-4]], r) for r in nq if r["id"][:-4] in old]
    ab = [(a, b_) for a, b_ in pr5 if a["answer"] == "Q"]; rest = [(a, b_) for a, b_ in pr5 if a["answer"] != "Q"]
    N.update(RFiveTrWithQ=frac(sum(a["correct"] for a, _ in pr5), len(pr5)),
             RFiveTrForced=frac(sum(b_["correct"] for _, b_ in pr5), len(pr5)),
             RFiveTrAbstRight=frac(sum(b_["correct"] for _, b_ in ab), len(ab)),
             RFiveTrRestWithQ=frac(sum(a["correct"] for a, _ in rest), len(rest)),
             RFiveTrRestForced=frac(sum(b_["correct"] for _, b_ in rest), len(rest)))
    # costs
    allr = RES + PILOT + res5
    N.update(RFiveCalls=str(len(res5)), RFiveCost=f"\\${sum(r['cost'] for r in res5):.2f}",
             RFiveTotalCalls=f"{len(allr):,}", RFiveTotalCost=f"\\${sum(r['cost'] for r in allr):.2f}",
             RFiveMainCalls=str(sum(r.get("model") != "haiku" for r in allr)))
    return dict(trunks=t5, son_cue=sel("R6_sonnet_medium_6k", "MONO-CUE-214"),
                opus_cue=sel("R6_opus_low_6k", "MONO-CUE-214"), opus_neg_mono=opus_neg)


R5 = round5()

# ---------------- write ----------------
(HERE / "numbers.json").write_text(json.dumps(N, indent=1), encoding="utf-8")
with (HERE / "numbers.tex").open("w", encoding="utf-8") as f:
    f.write("% Generated by analysis.py from the saved replies. Do not edit by hand.\n")
    for k_, v in N.items():
        assert re.fullmatch(r"[A-Za-z]+", k_), k_
        f.write(f"\\newcommand{{\\n{k_}}}{{{v}}}\n")
print(f"{len(N)} numbers written")

# ---------------- figures ----------------
# Both figures are drawn on a canvas measured in points (1 unit = 1 pt, 468 pt wide), so every position is exact.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                     "lines.scale_dashes": False, "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt",
                     "pdf.fonttype": 42})
BLUE, VERM, WHITE = "#0072B2", "#D55E00", "#ffffff"          # Okabe-Ito blue and vermillion
GRID, SEP, AXIS, NOTE, HEAD, ROW, KEY = "#e4e4e4", "#d2d2d2", "#333333", "#666666", "#111111", "#333333", "#1a1a1a"
CHANCE = dict(color="#808080", lw=0.9, dashes=(2.4, 1.8))
CEIL = dict(color="#262626", lw=1.1, dashes=(5, 2))
FW, X0, X1 = 468.0, 137.35, 438.51                         # canvas width; x of 0% and of 100%
STYLE = {"son": (BLUE, "circle", True), "opus": (VERM, "diamond", True), "forced": (VERM, "diamond", False)}
NAME = {"son": "Sonnet 5.5 (medium effort)", "opus": "Opus 5.5 (low effort)",
        "forced": "Opus 5.5 (low effort), forced choice"}


def xp(p): return X0 + (X1 - X0) * p


def canvas(h):
    fig = plt.figure(figsize=(FW / 72, h / 72))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(h, 0); ax.axis("off")
    return fig, ax


def label(ax, x, yc, s, size, color, **kw):        # text vertically centred on yc
    return ax.text(x, yc + 0.34 * size, s, fontsize=size, color=color, va="baseline", **kw)


def seg(ax, x0, y0, x1, y1, **kw): ax.plot([x0, x1], [y0, y1], **kw)


def marker(ax, x, y, cond):
    col, shape, filled = STYLE[cond]
    if shape == "circle":
        ax.add_patch(Circle((x, y), 3.1, fc=col, ec=WHITE, lw=0.8, zorder=4)); return
    h = 4.1 if filled else 3.6
    ax.add_patch(Polygon([(x, y - h), (x + h, y), (x, y + h), (x - h, y)], fc=col if filled else WHITE,
                         ec=WHITE if filled else col, lw=0.8 if filled else 1.1, zorder=4))


def estimate(ax, k, n, lo, hi, y, cond):            # point, 90% interval with whiskers, and the count on the right
    col = STYLE[cond][0]
    if hi > lo:
        seg(ax, xp(lo), y, xp(hi), y, color=col, lw=0.9, zorder=3)
        for x_ in (xp(lo), xp(hi)): seg(ax, x_, y - 2.2, x_, y + 2.2, color=col, lw=0.9, zorder=3)
    marker(ax, xp(k / n), y, cond)
    label(ax, X1 + 8, y, f"{k}/{n}", 7.2, NOTE)


def frame(ax, top, bottom, xlabel):                 # vertical gridlines and the x axis
    for p in (0, 0.25, 0.5, 0.75, 1): seg(ax, xp(p), top, xp(p), bottom, color=GRID, lw=0.6, zorder=0)
    seg(ax, X0 - 0.35, bottom, X1 + 0.35, bottom, color=AXIS, lw=0.7)
    for p in (0, 0.25, 0.5, 0.75, 1):
        seg(ax, xp(p), bottom, xp(p), bottom + 3, color=AXIS, lw=0.7)
        ax.text(xp(p), bottom + 10.68, f"{round(100 * p)}%", fontsize=7.2, color=AXIS, ha="center", va="baseline")
    ax.text(xp(0.5), bottom + 22.08, xlabel, fontsize=8, color=KEY, ha="center", va="baseline")


def key(fig, ax, y, items):                         # one legend row; items are model keys or (line style, label)
    x, inv, r = 0.0, ax.transData.inverted(), fig.canvas.get_renderer()
    for it in items:
        if isinstance(it, str):
            half = 3.1 if STYLE[it][1] == "circle" else (4.1 if STYLE[it][2] else 3.6)
            marker(ax, x + half, y, it); t = label(ax, x + 2 * half + 4.4, y, NAME[it], 8, KEY)
        else:
            seg(ax, x, y, x + 15, y, **it[0]); t = label(ax, x + 19, y, it[1], 8, KEY)
        x = inv.transform((t.get_window_extent(renderer=r).x1, 0))[0] + 8.4


# Figure 1: polarity by gate family, bias cue and model
rows_s, rows_o = son, opl
rt_s, wr_s = cue_split(rows_s); rt_o, wr_o = cue_split(rows_o)
groups = [("AND, OR, MAJORITY-type gates",
           [("Trained as usual", [r for r in rows_s if fam(r) == "mono"], R5["opus_neg_mono"]),
            ("Bias sign points the wrong way", R5["son_cue"], R5["opus_cue"])]),
          ("XOR and parity",
           [("All networks", [r for r in rows_s if fam(r) == "xl"], [r for r in rows_o if fam(r) == "xl"]),
            ("Bias sign points the right way", rt_s, rt_o),
            ("Bias sign points the wrong way", wr_s, wr_o)])]
fig, ax = canvas(208.68)
frame(ax, 22.2, 183.2, "Output polarity right"); seg(ax, xp(0.5), 22.2, xp(0.5), 183.2, zorder=1, **CHANCE)
y = 30.2
for gi, (gname, rows) in enumerate(groups):
    y += 27.0 if gi else 0.0
    label(ax, 0, y, gname, 8, HEAD, fontweight="bold"); y += 20.0
    for ri, (lab, rs, ro) in enumerate(rows):
        y += 24.0 if ri else 0.0
        label(ax, 9, y, lab, 8, ROW)
        for cond, rr, dy in (("son", rs, -4.5), ("opus", ro, 4.5)):
            estimate(ax, sum(map(pol, rr)), len(rr), *cluster_ci(rr, "network"), y + dy, cond)
key(fig, ax, 5.6, ["son", "opus", (CHANCE, "chance")])
fig.savefig(HERE / "figures" / "fig_polarity.pdf"); fig.savefig(HERE / "figures" / "fig_polarity.png", dpi=200)

# Figure 2: shared trunks, by level
levels = [("Exact function", "chance 6.25%, first-layer ceiling 25%", lambda t: (t["exact"], t["N"]), 1 / 16, 0.25),
          ("Input group", "chance 50%", lambda t: (t["grp"], t["N"]), 0.5, None),
          ("Pair, given the right group", "chance 25%", lambda t: (t["pair"], t["grp"]), 0.25, None),
          ("Polarity, given the right pair", "chance 50%", lambda t: (t["polpair"], t["pair"]), 0.5, None)]
def level_rows(t, i):
    rows = t["rows"]
    if i == 0: return rows, lambda r: r["correct"]
    g = [r for r in rows if r["answer"] in GROUP and GROUP[r["answer"]] == GROUP[r["label"]]]
    if i == 1: return rows, lambda r: r["answer"] in GROUP and GROUP[r["answer"]] == GROUP[r["label"]]
    if i == 2: return g, lambda r: PAIR[r["answer"]] == PAIR[r["label"]]
    return [r for r in g if PAIR[r["answer"]] == PAIR[r["label"]]], lambda r: r["correct"]
C0, STEP = 54.3, 37.0
fig, ax = canvas(209.28)
frame(ax, C0 - STEP / 2, C0 + STEP * (len(levels) - 0.5), "Share right")
for i, (name, sub, f_, ch, ceil) in enumerate(levels):
    c = C0 + STEP * i
    if i: seg(ax, 0, c - STEP / 2, FW, c - STEP / 2, color=SEP, lw=0.6, zorder=0)
    ax.text(0, c - 2.61, name, fontsize=8, color=HEAD, va="baseline")
    ax.text(0, c + 7.27, sub, fontsize=7.2, color=NOTE, va="baseline")
    seg(ax, xp(ch), c - 16, xp(ch), c + 16, zorder=1, **CHANCE)
    if ceil: seg(ax, xp(ceil), c - 16, xp(ceil), c + 16, zorder=1, **CEIL)
    for cond, t, dy in (("son", TR["Son"], -9.5), ("opus", TR["Opus"], 0.0), ("forced", R5["trunks"], 9.5)):
        rr, st_ = level_rows(t, i); x_, n_ = f_(t)
        estimate(ax, x_, n_, *cluster_ci(rr, "network", stat=st_), c + dy, cond)
key(fig, ax, 5.6, ["son", "opus", "forced", (CHANCE, "chance")])
key(fig, ax, 19.2, [(CEIL, "first-layer ceiling")])
fig.savefig(HERE / "figures" / "fig_trunks.pdf"); fig.savefig(HERE / "figures" / "fig_trunks.png", dpi=200)
print("figures written")
