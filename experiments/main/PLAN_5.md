# Round 5: two pre-registered questions (plan and pre-registration)

Written Nov 5, 2026, after round 4, before any round-5 call. Spend so far: $39.13 in this folder, $43.37 across all API work. Hard cap $15.00, enforced in `run_round5.py`. Expected about $14.

## Why
- **G1 and G2: is monotone polarity read, or taken from the bias sign?** Sonnet got output polarity right on 48/48 monotone-gate stimuli (AND, OR, AND-NOT, MAJORITY, 3-AND, 3-OR and their NOTs). But on 46 of the 48 the output-bias sign already points the right way, so reading the weights and following the bias sign give the same answer. The other two are one network (`P2_D_H214`) with b2 = +0.065 and only 45% of W2 positive, so its cue is barely reversed. Opus has never been run at low effort on monotone gates.
- **G3: did abstaining hide decoding?** On the 64 rescaled trunk readouts, Opus answered "none of the above" 15 times and was right on 21 of the other 49. Its per-trunk accuracy (33%, 90% CI 22–44%) left finding 4 inconclusive. Forced choice asks whether it decodes readouts when it can't abstain.

## Stimuli (`make_round5.py`, free; 124 stimuli; every label checked on the weights as displayed)
**MONO-CUE-214 (24).** 12 new single-task 214-unit networks, 2 per monotone plain function (A, B, D, I, J, K). Each has the bias cue reversed by the CUE-214 definition: b2 > 0 and more than half of W2 positive, where a plain function needs b2 < 0. Each is shown as trained and fully negated (−W2, −b2), so all 24 have `bias_cue_ok = False`. Seeds are new, from 900000.

Round 3 found no such networks in 12 seeds per function, so for each function both networks come from the first method, in this order, that gives two qualifying networks:
- **(a)** Normal training (`make_round3.train_single`), 1,000 seeds per function.
- **(b)** The same training with the output bias held at +c (initialised at c, zero gradient), 200 seeds per function. Here c is the function's mean |b2| over its three NEG-214 networks.
- **(c)** An exact shift through always-active units, applied to a network from (a) that already has more than half of W2 positive. Every hidden unit active on all 4,096 inputs with W2_j < 0 gets the same increase δ in b1_j, and b2 gets +δ·Σ|W2_j|, with δ set so that b2 = +c. Each such unit is linear, so the function is unchanged before rounding.
- **(d)** Added on Nov 5, before any call, after (a)–(c) failed for A and J; this was the user's decision. Apply the shift of (c) to a network from (a). Then flip the sign of the negative output weights of the fewest dead hidden units (never active on any input, smallest |W2_j| first) until more than half of W2 is positive. This is exact, because a dead unit never contributes.

The table shows how many networks qualified under each method, and which method each function used. A dash means the method wasn't tried, because an earlier one had already worked.

| Function | (a) 1,000 seeds | (b) 200 seeds | (c) | (d) | Used | Seeds | Edit size |
|---|---|---|---|---|---|---|---|
| A (AND) | 0 | 0 | 0 | 41 | (d) | 900005, 900019 | shift on 6 and 9 units (δ = 0.63, 0.40); 15 and 8 dead units flipped |
| B (OR) | 0 | 102 | – | – | (b) | 970000, 970001 | b2 held at 0.831 |
| D (AND-NOT) | 41 | – | – | – | (a) | 920018, 920024 | none |
| I (MAJORITY) | 0 | 0 | 11 | – | (c) | 930018, 930032 | shift on 6 and 4 units (δ = 0.61, 0.92) |
| J (3-AND) | 0 | 0 | 0 | 8 | (d) | 940003, 940004 | shift on 13 and 8 units (δ = 0.59, 0.88); 27 and 18 dead units flipped |
| K (3-OR) | 1 | 96 | – | – | (b) | 1010000, 1010001 | b2 held at 1.307 |

Under normal training, b2 > 0 never occurred in 1,000 seeds for A, B, I or J. K's one hit has b2 = +0.013. For A and J, no network from (a) or (b) has more than half of W2 positive: the maximum share is 48% and 45% under (a), and lower under (b). (c) changes only b1 and b2, so it can't raise the share either.

The negated networks' labels (the complement) are checked as well.

**Weight statistics.** Each value is a mean over the networks as trained. NEG-214 means the same function's three NEG-214 networks. "Always" and "dead" count the hidden units active on all 4,096 inputs and on none.

| | mean \|W1\| | mean \|b1\| | mean \|W2\| | \|b2\| | max \|b1\| | always | dead | W2 > 0 |
|---|---|---|---|---|---|---|---|---|
| A NEG-214 / MONO-CUE | 0.128 / 0.127 | 0.211 / 0.212 | 0.397 / 0.384 | 2.10 / 2.10 | 1.31 / **1.67** | 11.7 / 9.0 | 24.0 / 35.0 | 42% / 51% |
| B | 0.107 / 0.114 | 0.111 / 0.117 | 0.289 / 0.293 | 0.83 / 0.83 | 0.65 / 0.65 | 7.3 / 6.5 | 11.0 / 20.5 | 47% / 55% |
| D | 0.112 / 0.105 | 0.120 / 0.118 | 0.314 / 0.269 | 0.16 / 0.10 | 0.59 / 0.50 | 4.0 / 4.5 | 13.0 / 11.0 | 46% / 51% |
| I | 0.131 / 0.143 | 0.205 / 0.227 | 0.392 / 0.404 | 2.24 / 2.24 | 1.22 / **1.92** | 7.3 / 7.5 | 20.7 / 26.5 | 40% / 51% |
| J | 0.161 / 0.168 | 0.299 / 0.338 | 0.505 / 0.552 | 2.97 / 2.97 | 2.17 / 2.28 | 10.0 / 11.0 | 42.7 / 47.5 | 36% / 51% |
| K | 0.124 / 0.118 | 0.129 / 0.131 | 0.311 / 0.331 | 1.31 / 1.31 | 0.71 / 0.61 | 8.0 / 6.5 | 9.7 / 22.5 | 45% / 54% |
| All six | 0.127 / 0.129 | 0.179 / 0.190 | 0.368 / 0.372 | 1.60 / 1.59 | 1.11 / 1.27 | 8.1 / 7.5 | 20.2 / 27.2 | 43% / 52% |

**Artifacts, stated in advance:**
- **Visible: a larger max |b1| where the shift was used.** I: 1.92 vs 1.22; A: 1.67 vs 1.31. J's 2.28 is within its NEG-214 range of 1.71–2.42. The raised units are always-active units with negative output weights, so each adds a large negative term at the all-zero input, the term that offsets the positive b2. A reader that evaluates the largest terms at x = 0 can find polarity there; a reader that uses only the bias sign cannot. This could make polarity easier on A, I and J than on networks where the offset is spread over many units.
- **Not visible without computation: more dead units.** All six: 27.2 vs 20.2; B and K, made by (b): about twice as many. In (d), 8–27 dead units also have their output weight's sign flipped. A dead unit contributes nothing, and seeing that it is dead requires checking its maximum pre-activation.
- **|b2| matches by design.** For A, B, I, J and K it equals the function's NEG-214 mean, by construction. D's b2 is natural, +0.10 vs a NEG-214 mean |b2| of 0.16.

Function and construction method coincide, so a per-function difference can't be separated from a per-method one. Both are reported descriptively.

**NEG-214 monotone (36).** Round 3's NEG-214 stimuli of the six monotone functions, copied unchanged: 18 networks, original and negated. On 34 of them the bias cue points the right way. These give Opus's cue-right baseline.

**TRUNKR-214-NOQ (64).** Round 4's 64 TRUNKR-214 stimuli. Only the prompt changes:
- the line "Q: none of the above (the network computes some other function)" is removed;
- ", including Q" is removed from "Any option may be correct, including Q.";
- the answer instruction asks for "one letter from A to P".

Nothing else changes; `test_round5.py` checks that undoing the three edits restores round 4's prompt exactly. The ids get the suffix `_noq`, and `src` points to the round-4 stimulus.

## Runs, in priority order (`run_round5.py`; run names `R6_*` continue round 4's `R5_*`)
| Step | Model, effort, cap | Stimuli | Calls | Worst case | Expected |
|---|---|---|---|---|---|
| 1 | Sonnet 5.5 (`claude-sonnet-5-5`), medium, 6k | MONO-CUE-214 | 24 | $2.35 | ~$1.15 |
| 2 | Opus 5.5 (`claude-opus-5-5`), low, 6k | MONO-CUE-214 | 24 | $4.70 | ~$2.35 |
| 3 | Opus low, 6k | NEG-214 monotone | 36 | $7.04 | ~$3.50 |
| 4 | Opus low, 6k | TRUNKR-214-NOQ | 64 | $12.53 | ~$7.15 |
| | | | 148 | $26.61 | ~$14.15 |

Expected cost is from round 4's mean cost per call: Sonnet $0.048 on single-task networks; Opus $0.097 on single-task networks and $0.112 on rescaled trunks.

**Budget guard.** The worst case (input estimate plus the full 6k output cap) is reserved before each call and settled after, as in rounds 3–4. Two changes from the base runner keep a round that ends near the cap from losing calls it could afford:
- Requests are served in order and wait for in-flight calls to settle rather than being skipped.
- Once a call cannot fit even with nothing in flight, it and every later call are skipped.

So the cap is never exceeded, and calls are dropped only from the end of the priority order. Steps 1–3 cost at most $14.08 even if every call hits its cap, so they always finish. Step 4 gets the rest. If it is cut, the last trunks are lost, and G3 uses complete trunks only (all 4 readouts answered or truncated); the number of trunks is reported.

## Pre-registered rules
A truncated, missing or "Q" answer counts as wrong. Polarity is right when the answer is one of A–P and is a plain option exactly when the label is (`pol_ok`). The pair is right when the answer is the label or its complement (the function ignoring polarity).

**G1 (Sonnet) and G2 (Opus): monotone polarity vs the bias sign** (confirmatory). For each model, compare the polarity-right rate on MONO-CUE-214 (24) with the same model's cue-right monotone stimuli:
- Sonnet: its round-3 run `R1_sonnet_medium_6k` on the NEG-214 and NATIVE-214 monotone stimuli with the cue pointing the right way (46, all polarity-right).
- Opus: the cue-right subset of run 3 (34 of the 36 NEG-214 monotone stimuli).

Drop = baseline rate − MONO-CUE rate. The test is one-sided Fisher exact (baseline > MONO-CUE). The rules are applied in this order:
1. Drop ≥ 30 points, Fisher p < .05, and pair accuracy on MONO-CUE ≥ 60% → **"monotone polarity follows the bias sign"**.
2. Polarity ≥ 85% on MONO-CUE → **"computes monotone polarity; the shortcut is XOR/parity-specific"**.
3. Drop ≥ 30 points and p < .05, but pair accuracy < 60% → **"inconclusive: the new networks are harder overall"**.
4. Otherwise → **inconclusive**.

Rules 1 and 2 can't both hold, since a 30-point drop leaves at most 70%. Also reported:
- exact-gate accuracy;
- a one-sided network-level permutation test (network labels permuted between baseline and MONO-CUE, 200,000 permutations, statistic = difference in mean polarity-right count per network);
- how many wrong answers are the right pair with flipped polarity, the shortcut's signature;
- the split by version (original/negated) and by function. Function and construction method coincide here (one method per function), so the split by method is descriptive.

**G3: Opus readout decoding under forced choice** (confirmatory). Same as F4a. For each complete trunk, take exact accuracy minus 0.25 (the ceiling for a reader that uses only the first layer), and run a one-sided exact sign-flip test over the 16 trunks.
- p < .05 → **"Opus decodes readouts under forced choice"**.
- Otherwise, if the upper 90% bootstrap bound on mean per-trunk accuracy is below 40% → **"no decoding even under forced choice"**.
- Otherwise inconclusive.

This is reported separately from the original F4a, which is not re-decided, with the same breakdown: input group, pair given group, and polarity given pair.
- **Exploratory:** the same sign-flip test against 12.5%, the expectation for a reader with perfect polarity and nothing else.
- **Descriptive:** the same 64 readouts with Q allowed (round 4) vs forced choice, and forced-choice accuracy on the 15 readouts answered Q in round 4.

## Files
- `make_round5.py` builds `stimuli_round5.jsonl` in about 45 minutes on 16 cores, with no API calls.
- `run_round5.py` reuses `run_round3.py` (call, helpers) and `run_round4.py` (trunk statistics). Use `--dry-run`, no flag to run, or `--report`. Replies go to `results_round5.jsonl` and the console log to `run_log_5.txt`.
- `test_round5.py` holds 10 tests. Run the whole suite with `python -m pytest test_round5.py test_round4.py test_round3.py test_round3b.py -q`.
- `PREREGISTRATION_5.txt` records the SHA-256 hashes and the UTC time. This plan, the code and the stimuli are committed and pushed to GitHub before the first round-5 call.
