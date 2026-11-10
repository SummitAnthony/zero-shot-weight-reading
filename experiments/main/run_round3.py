#!/usr/bin/env python3
"""Round-3 runner (design and pre-registered decision rules: PLAN.md).

Same system prompt, answer-instruction style, budget guard and API call as 07_API_Pilot; answer letters A-Q.
Hard cap: BUDGET_USD of spend recorded in results_round3.jsonl. Before each call the worst case (input estimate
plus the full output cap) is reserved; the call is skipped if that could pass the cap. Steps run strictly in
priority order, so a budget cut removes the last steps.

  python run_round3.py --dry-run     # free: call plan and worst-case cost
  python run_round3.py               # runs the calls; resumable
  python run_round3.py --report      # tables and decision rules D1-D6
"""
import argparse, itertools, json, math, os, re, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from math import comb
from pathlib import Path

BUDGET_USD = 13.00
WORKERS = 4
HERE = Path(__file__).resolve().parent
STIMULI = HERE / "stimuli_round3.jsonl"
RESULTS = HERE / "results_round3.jsonl"
MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
PRICE = {"haiku": (0.10, 0.50), "sonnet": (2.00, 10.00), "opus": (4.00, 20.00)}   # standard USD / M tokens, Nov 4 2026
SYSTEM = "You are a careful analyst. Answer the question about the network you are given."   # as in 07_API_Pilot
PLAIN = set("ABCDIJKL")
GROUP = {**{k: "x2,x5" for k in "ABCDEFGH"}, **{k: "x0,x3,x7" for k in "IJKLMNOP"}}
PAIR = {k: frozenset(p) for p in ("AE", "BF", "CG", "DH", "IM", "JN", "KO", "LP") for k in p}

P2_214 = lambda s: s["set"] == "NEG-214" and s["network"].startswith("P2_")
SET = lambda name: (lambda s: s["set"] == name)
TRUNK_HALF = lambda s: s["network"] in ("T0_H214", "T1_H214", "T2_H214", "T3_H214")
# (run name, model, effort, max_tokens, warned, stimulus filter), in priority order. One step per row.
STEPS = [
    ("R1_sonnet_medium_6k", "sonnet", "medium", 6000, False, SET("NEG-214")),
    ("R1_sonnet_medium_6k", "sonnet", "medium", 6000, False, SET("TRUNK-214")),
    ("R2_haiku_low_6k", "haiku", "low", 6000, False, P2_214),
    ("R2_haiku_medium_6k", "haiku", "medium", 6000, False, P2_214),
    ("R3_sonnet_medium_6k_warned", "sonnet", "medium", 6000, True, SET("NEG-214")),
    ("R1_sonnet_medium_6k", "sonnet", "medium", 6000, False, SET("CUE-214")),
    ("R1_sonnet_medium_6k", "sonnet", "medium", 6000, False, SET("NATIVE-214")),
    ("R3_sonnet_medium_6k_warned", "sonnet", "medium", 6000, True, TRUNK_HALF),
    ("R4_opus_medium_8k", "opus", "medium", 8000, False, P2_214),
    ("R1_sonnet_medium_6k", "sonnet", "medium", 6000, False, SET("NEG-36")),
]


def load():
    return [json.loads(l) for l in STIMULI.read_text(encoding="utf-8").splitlines() if l.strip()]


def plan(stim):
    return [dict(step=i, run=run, model=model, effort=effort, max_tokens=mt, warned=warned, s=s, sample=0)
            for i, (run, model, effort, mt, warned, flt) in enumerate(STEPS) for s in stim if flt(s)]


class Budget:   # as in 07_API_Pilot
    def __init__(self, cap, spent):
        self.cap, self.spent, self.reserved, self.lock = cap, spent, 0.0, threading.Lock()
    def reserve(self, amount):
        with self.lock:
            if self.spent + self.reserved + amount > self.cap: return False
            self.reserved += amount; return True
    def settle(self, reserved, actual):
        with self.lock:
            self.reserved -= reserved; self.spent += actual


def cost(model, tin, tout):
    pi, po = PRICE[model]; return (tin * pi + tout * po) / 1e6


def est_tokens(text): return int(len(text) * 0.75) + 50     # measured ~0.70 tokens per character on these prompts


def make_client():
    try:
        import anthropic
    except ImportError:
        sys.exit("Install the SDK first:  pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        env = HERE.parent / ".env"          # the project's key file (gitignored); read without printing it
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY=") and len(line) > 20:
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
    if not os.environ.get("ANTHROPIC_API_KEY"): sys.exit("Set ANTHROPIC_API_KEY first.")
    return anthropic.Anthropic(max_retries=5, timeout=900)


def parse_answer(text):
    m = re.findall(r"ANSWER:\s*\**\s*([A-Q])\b", text)
    return m[-1] if m else None


def call(client, job):
    t0 = time.time(); s = job["s"]
    r = client.messages.create(model=MODELS[job["model"]], max_tokens=job["max_tokens"], system=SYSTEM,
                               messages=[{"role": "user", "content": s["prompt_warned"] if job["warned"] else s["prompt"]}],
                               extra_body={"output_config": {"effort": job["effort"]}})
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return dict(answer=parse_answer(text), stop_reason=r.stop_reason, model_id=r.model, input_tokens=r.usage.input_tokens,
                output_tokens=r.usage.output_tokens, seconds=round(time.time() - t0, 1), text=text)


def rows():
    return [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines()] if RESULTS.exists() else []


def run(args):
    stim = load(); jobs = plan(stim); done = rows()
    keys = {(r["run"], r["id"], r["sample"]) for r in done}; spent = sum(r["cost"] for r in done)
    todo = [j for j in jobs if (j["run"], j["s"]["id"], j["sample"]) not in keys]
    worst = lambda j: cost(j["model"], est_tokens(j["s"]["prompt_warned"]), j["max_tokens"])
    print(f"{len(jobs)} calls planned, {len(todo)} to go. Spent ${spent:.2f}. Worst case for what is left "
          f"${sum(map(worst, todo)):.2f}; cap ${BUDGET_USD:.2f}.")
    if args.dry_run:
        acc = 0.0
        for i, (run_name, model, *_rest) in enumerate(STEPS):
            js = [j for j in jobs if j["step"] == i]; w = sum(map(worst, js)); acc += w
            print(f"  step {i + 1:2d} {run_name:28s} {js[0]['s']['set'] if js else '':10s} {len(js):3d} calls  "
                  f"worst ${w:5.2f}  cumulative worst ${acc:6.2f}")
        print("Dry run only. Nothing was sent."); return
    client = make_client(); budget = Budget(BUDGET_USD, spent); lock = threading.Lock(); skipped = 0

    def work(j):
        w = worst(j)
        if not budget.reserve(w): return "skipped"
        try:
            r = call(client, j)
        except Exception as e:
            budget.settle(w, 0.0); return f"error: {e}"
        c = cost(j["model"], r["input_tokens"], r["output_tokens"]); budget.settle(w, c); s = j["s"]
        row = dict(run=j["run"], step=j["step"], model=j["model"], effort=j["effort"], max_tokens=j["max_tokens"],
                   warned=j["warned"], id=s["id"], sample=j["sample"], set=s["set"], network=s["network"], H=s["H"],
                   nweights=s["nweights"], version=s["version"], func=s["func"], label=s["label"],
                   bias_cue_ok=s["bias_cue_ok"], correct=r["answer"] == s["label"], cost=round(c, 5),
                   utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), **r)
        with lock:
            with RESULTS.open("a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
        return (f"step {j['step'] + 1} {j['run']} {s['id']}: {r['answer']} (truth {s['label']}) out={r['output_tokens']} "
                f"{r['stop_reason']} ${c:.3f} | total ${budget.spent:.2f}")

    for i in range(len(STEPS)):
        batch = [j for j in todo if j["step"] == i]
        if not batch: continue
        with ThreadPoolExecutor(WORKERS) as ex:
            for fut in as_completed([ex.submit(work, j) for j in batch]):
                msg = fut.result()
                if msg == "skipped": skipped += 1
                else: print(msg, flush=True)
    if skipped: print(f"{skipped} calls skipped to stay under the ${BUDGET_USD:.2f} cap.")
    report()


# ---------------- statistics ----------------
def mcnemar_one_sided(b, c):
    n = b + c
    return 1.0 if n == 0 else sum(comb(n, i) for i in range(b, n + 1)) / 2 ** n


def fisher_one_sided(x1, n1, x2, n2):
    """P(group 1 has >= x1 successes | margins): evidence that group 1's rate exceeds group 2's."""
    K, N = x1 + x2, n1 + n2
    return sum(comb(K, i) * comb(N - K, n1 - i) for i in range(x1, min(K, n1) + 1)) / comb(N, n1)


def tango_ci(b, c, n, conf=0.90):
    """Tango (1998) score interval for a paired difference (b - c) / n; b, c are the two discordant counts."""
    z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054}[conf]

    def Z(d):
        A, B, C = 2 * n, -b - c + (2 * n - b + c) * d, -c * d * (1 - d)
        q = (-B + math.sqrt(max(B * B - 4 * A * C, 0.0))) / (2 * A)
        var = n * (2 * q + d - d * d)
        num = b - c - n * d
        return 0.0 if num == 0 else (math.copysign(math.inf, num) if var <= 0 else num / math.sqrt(var))

    def root(f, lo, hi):        # f decreasing; returns x with f(x) ~ 0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        return (lo + hi) / 2

    d = (b - c) / n; eps = 1e-9
    lo = -1.0 if Z(-1 + eps) < z else root(lambda x: Z(x) - z, -1 + eps, d)
    hi = 1.0 if Z(1 - eps) > -z else root(lambda x: Z(x) + z, d, 1 - eps)
    return lo, hi


def sign_flip_p(values):
    """Exact one-sided sign-flip permutation test that the mean of `values` (one per cluster) exceeds 0."""
    obs = sum(values); k = len(values)
    hits = sum(sum(s * v for s, v in zip(signs, values)) >= obs - 1e-12 for signs in itertools.product((1, -1), repeat=k))
    return hits / 2 ** k


# ---------------- analysis ----------------
def pct(x, n): return f"{x}/{n} ({x / n:.0%})" if n else "-"
def ci(lo, hi): return f"90% CI [{lo:+.0%}, {hi:+.0%}]"
def is_plain(a): return a in PLAIN
def pol_ok(r): return r["answer"] in GROUP and is_plain(r["answer"]) == is_plain(r["label"])
def trunc(r): return r["stop_reason"] == "max_tokens" or r["answer"] is None


def pairs_of(d, set_name, nets=None):
    out = []
    for n in sorted({r["network"] for r in d.values() if r["set"] == set_name}):
        if nets and not nets(n): continue
        o, g = d.get(f"{n}_orig"), d.get(f"{n}_neg")
        if o and g: out.append((o, g))
    return out


def polarity(tag, pairs):
    """Prints originals / negated accuracy and polarity sensitivity; returns (lo, hi) of the sensitivity CI."""
    k = len(pairs)
    oc = sum(o["correct"] for o, _ in pairs); nc = sum(g["correct"] for _, g in pairs)
    both = sum(o["correct"] and g["correct"] for o, g in pairs); named = sum(g["answer"] == g["func"] for _, g in pairs)
    b = sum(is_plain(o["answer"]) and not is_plain(g["answer"]) for o, g in pairs)
    c = sum(not is_plain(o["answer"]) and is_plain(g["answer"]) for o, g in pairs)
    lo, hi = tango_ci(b, c, k)
    print(f"{tag}: originals {pct(oc, k)}, negated {pct(nc, k)}, both right {pct(both, k)}, negated named as the "
          f"original {pct(named, k)}.\n    Polarity sensitivity P(plain|orig) - P(plain|neg) = {(b - c) / k:+.0%} "
          f"({ci(lo, hi)}; b={b}, c={c}, one-sided McNemar p={mcnemar_one_sided(b, c):.4f})")
    return lo, hi


def report():
    R = rows()
    if not R: print("No results yet."); return
    print(f"\n=== Round 3: {len(R)} calls, ${sum(r['cost'] for r in R):.2f} spent ===")
    g = defaultdict(list)
    for r in R: g[(r["run"], r["set"], r["version"])].append(r)
    print(f"{'run':28s} {'set':10s} {'ver':6s} {'correct':>12s} {'polarity ok':>12s} {'group ok':>9s} {'Q':>3s} "
          f"{'trunc':>5s} {'mean out':>8s} {'max':>6s}")
    for (run, st, v), rs in sorted(g.items()):
        n = len(rs); outs = [r["output_tokens"] for r in rs]
        print(f"{run:28s} {st:10s} {v:6s} {pct(sum(r['correct'] for r in rs), n):>12s} {pct(sum(map(pol_ok, rs)), n):>12s} "
              f"{sum(r['answer'] in GROUP and GROUP[r['answer']] == GROUP[r['label']] for r in rs):9d} "
              f"{sum(r['answer'] == 'Q' for r in rs):3d} {sum(map(trunc, rs)):5d} {sum(outs) / n:8.0f} {max(outs):6d}")
    by = lambda run: {r["id"]: r for r in R if r["run"] == run}
    u, w, o4 = by("R1_sonnet_medium_6k"), by("R3_sonnet_medium_6k_warned"), by("R4_opus_medium_8k")

    # D1: polarity on a clean break
    p = pairs_of(u, "NEG-214")
    if p:
        print()
        lo, hi = polarity(f"D1 (Sonnet, NEG-214, {len(p)} pairs)", p)
        print("D1 verdict:", "POLARITY-SENSITIVE: uses the output layer for polarity" if lo > 0 else
              "OUTPUT-LAYER BLIND for polarity" if hi < 0.30 else "INCONCLUSIVE")
    xl = [r for r in u.values() if r["set"] in ("CUE-214", "NEG-214") and r["func"] in "CGLP" and r["version"] in ("orig", "neg")]
    if any(r["set"] == "CUE-214" for r in xl):
        right = [r for r in xl if r["bias_cue_ok"]]; wrong = [r for r in xl if not r["bias_cue_ok"]]
        a, b_ = sum(map(pol_ok, right)), sum(map(pol_ok, wrong))
        pf = fisher_one_sided(a, len(right), b_, len(wrong))
        gap = a / max(1, len(right)) - b_ / max(1, len(wrong))
        print(f"D1-cue (Sonnet, XOR/parity networks): polarity right when the bias cue points the right way "
              f"{pct(a, len(right))}, the wrong way {pct(b_, len(wrong))}; gap {gap:+.0%}, one-sided Fisher p={pf:.4f} -> "
              + ("BIAS-SIGN SHORTCUT" if gap >= 0.30 and pf < 0.05 else "NO EVIDENCE OF A BIAS-SIGN SHORTCUT"))
    nat = pairs_of(u, "NATIVE-214")
    if nat:
        k = len(nat); po = sum(is_plain(o["answer"]) for o, _ in nat)
        print(f"D1-native (Sonnet, networks trained on NOT f, {k}): originals right {pct(sum(o['correct'] for o, _ in nat), k)}, "
              f"answered a plain option {pct(po, k)}; their negations (plain f) right {pct(sum(g['correct'] for _, g in nat), k)}")
        polarity("    D1-native polarity", nat)

    # D2: trunks
    t = [r for r in u.values() if r["set"] == "TRUNK-214"]
    if t:
        per = defaultdict(list)
        for r in t: per[r["network"]].append(r)
        accs = [sum(r["correct"] for r in rs) / len(rs) for rs in per.values()]
        pf = sign_flip_p([a - 0.25 for a in accs])
        grp = [r for r in t if r["answer"] in GROUP and GROUP[r["answer"]] == GROUP[r["label"]]]
        pr = [r for r in grp if PAIR[r["answer"]] == PAIR[r["label"]]]
        distinct = sum(len({r["answer"] for r in rs}) for rs in per.values()) / len(per)
        print(f"\nD2 (Sonnet, TRUNK-214, {len(per)} trunks x 4): exact {pct(sum(r['correct'] for r in t), len(t))} "
              f"(a first-layer-only reader is capped at 25% per trunk); per trunk "
              f"{', '.join(f'{a:.0%}' for a in accs)}; sign-flip test over trunks p={pf:.4f}")
        print(f"    input group right {pct(len(grp), len(t))} (chance 50%); pair right given group {pct(len(pr), len(grp))} "
              f"(chance 25%); polarity right given pair {pct(sum(r['correct'] for r in pr), len(pr))} (chance 50%); "
              f"distinct answers per trunk {distinct:.1f} of 4")
        print("D2 verdict:", "READS THE OUTPUT LAYER when the first layer can't tell" if pf < 0.05 else
              "NO EVIDENCE it reads the output layer when the first layer can't tell")
        if p:
            oc = sum(o["correct"] for o, _ in p); tc = sum(r["correct"] for r in t)
            print(f"    Descriptive (scale-confounded): originals {pct(oc, len(p))} vs trunk {pct(tc, len(t))}, "
                  f"one-sided Fisher p={fisher_one_sided(oc, len(p), tc, len(t)):.4f}")

    # D3: warned vs unwarned
    if w:
        neg = [(u[i], w[i]) for i in w if i in u and w[i]["set"] == "NEG-214" and w[i]["version"] == "neg"]
        org = [(u[i], w[i]) for i in w if i in u and w[i]["set"] == "NEG-214" and w[i]["version"] == "orig"]
        if neg:
            k = len(neg); b = sum(bb["correct"] and not a["correct"] for a, bb in neg); c = sum(a["correct"] and not bb["correct"] for a, bb in neg)
            lo, hi = tango_ci(b, c, k); gain = (b - c) / k
            drop = (sum(a["correct"] for a, _ in org) - sum(bb["correct"] for _, bb in org)) / max(1, len(org))
            print(f"\nD3 (Sonnet, NEG-214 negated, {k}): unwarned {pct(sum(a['correct'] for a, _ in neg), k)}, warned "
                  f"{pct(sum(bb['correct'] for _, bb in neg), k)}; gain {gain:+.0%} ({ci(lo, hi)}). Warned originals: "
                  f"{pct(sum(bb['correct'] for _, bb in org), len(org))} vs unwarned {pct(sum(a['correct'] for a, _ in org), len(org))} "
                  f"(drop {drop:+.0%})")
            print("D3 verdict:", "SHIFT TOWARD NOT ANSWERS, not reading (warned originals drop > 10 points)" if drop > 0.10 else
                  "DEFAULT STRATEGY: the ability is there" if lo > 0 else "NO LARGE GAIN from the warning" if hi < 0.20 else
                  "INCONCLUSIVE")
            wp = pairs_of(w, "NEG-214")
            if wp: polarity("    D3 polarity under the warning", wp)
        wt = [(u[i], w[i]) for i in w if i in u and w[i]["set"] == "TRUNK-214"]
        if wt:
            print(f"    Warned trunk (T0-T3, {len(wt)}): unwarned {pct(sum(a['correct'] for a, _ in wt), len(wt))}, "
                  f"warned {pct(sum(bb['correct'] for _, bb in wt), len(wt))}")
    if o4:
        print()
        polarity(f"D4 Opus (NEG-214, Pilot 2 networks)", pairs_of(o4, "NEG-214"))
    for run in ("R2_haiku_low_6k", "R2_haiku_medium_6k"):
        h = list(by(run).values())
        if h:
            tr = sum(map(trunc, h)) / len(h); oc = sum(r["correct"] for r in h if r["version"] == "orig")
            print(f"D5 {run}: truncated {tr:.0%}, originals correct {oc}/8 -> "
                  f"{'USABLE' if tr <= 0.10 and oc >= 3 else 'NOT USABLE above 505 weights'}")
    p36 = pairs_of(u, "NEG-36")
    if p36: polarity(f"D6 (Sonnet, NEG-36, 505 weights, {len(p36)} pairs)", p36)

    if t:
        pl = sum(is_plain(r["answer"]) for r in t if r["answer"] in GROUP)
        print(f"\nPlain-option share of Sonnet's trunk answers (half the true labels are plain): "
              f"{pct(pl, sum(r['answer'] in GROUP for r in t))}")
    cue_re = re.compile(r"output bias|bias of the output|\bb2\b|out(put)? .{0,20}bias|all[- ]zeros?\b|x\s*=\s*0\b|all inputs (are )?(0|zero)", re.I)
    neg_rows = [r for r in R if r["model"] == "sonnet" and r["version"] in ("orig", "neg")]
    if neg_rows:
        print("Replies that mention the output bias or the all-zero input (Sonnet, orig/neg stimuli): " +
              ", ".join(f"{v} {pct(sum(bool(cue_re.search(r.get('text', ''))) for r in neg_rows if r['version'] == v), sum(r['version'] == v for r in neg_rows))}"
                        for v in ("orig", "neg")))
    intact = [r for r in R if r["version"] in ("orig", "trunk")]
    print(f"'None' (Q) answers on intact networks, all runs: {pct(sum(r['answer'] == 'Q' for r in intact), len(intact))}")
    print("Model IDs returned:", sorted({r["model_id"] for r in R}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a)
