# Results of rounds 3–5: what Claude reads in a network's weights

Nov 5, 2026. Total API spend for the project is $57.60: the API pilot (`07_`) cost $4.24, and this folder $53.36.

| Round | Calls | Cost | Committed before its first call |
|---|---|---|---|
| 3 | 248 | $12.10 | `a969b73` |
| 3b | 66 | $4.17 | `28de3fd` |
| 3c | 32 | $3.11 | `54ba5d2` |
| 4 | 240 | $19.76 | `3dbd427`, also pushed to GitHub before the run |
| 5 | 148 | $14.23 | `e74e9f5`, also pushed to GitHub before the run |

Rounds 3b, 3c, 4 and 5 are follow-ups motivated by earlier results and are labelled as such. Every number below can be reproduced from the saved replies with `python run_round3.py --report` (and the same for `run_round3b.py`, `run_round3c.py`, `run_round4.py` and `run_round5.py`).

## Bottom line
1. **The pilot's "readers ignore the output layer" came from an edit that leaves the output cues unchanged.** 14 of its 24 broken networks were dead, but that is not why Sonnet missed them: it caught 4/14 dead and 3/10 live ones (found in the paper review). The partial flip keeps the output bias's sign and the sign balance. On a clean break (full output negation, which computes exactly NOT f), Sonnet tracks the flipped output: polarity sensitivity +75%, 90% CI [+47%, +89%] on 24 pairs. It named the original function on a negated network only 2/24 times.
2. **Sonnet's output reading depends on the function.**
   - On AND, OR, AND-NOT, MAJORITY and their NOTs, polarity was right 48/48 at ~3,000 weights.
   - On XOR and parity it is near chance. Pooled over rounds 3–4, it gets 54/108 when the bias sign points the right way or the wrong way equally.
   - **Round 5:** the monotone polarity is computed, not read from the bias sign. On 24 cue-reversed monotone stimuli Sonnet's polarity stayed at 22/24, and Opus's at 24/24 (see below).
3. **Opus reads XOR/parity polarity that Sonnet misses.** *Replicated in round 4* on 24 new networks:
   - Opus at low effort: polarity 44/48 (92%), exact gate 38/48 (79%).
   - Sonnet at medium effort: polarity 25/48 (52%), exact gate 18/48 (38%).
   - Gap +40 points, exact sign-flip test over networks p < .0001.
   - Opus stays right whichever way the output bias points: 23/24 when it points the right way, 21/24 when it points the wrong way.
   - More thinking doesn't rescue Sonnet. At high effort with a 12k cap it ran out of tokens on 15 of 16 stimuli (mean 11,935 output tokens).
4. **Both models decode the output layer only partly.** On 16 shared trunks rescaled to single-task weight statistics:
   - Sonnet: 22% per trunk, 90% CI [14%, 31%]; p = .80 against the 25% ceiling for a reader that uses only the first layer.
   - Opus: 33%, CI [22%, 44%], p = .20 against that ceiling.
   - Under the pre-registered rule this is *inconclusive*, because Opus's upper bound (44%) doesn't exclude modest decoding.
   - The 25% is a ceiling, not an expectation. Chance is 6.25%, and a reader with perfect polarity but nothing else expects 12.5%. Against 12.5% (exploratory): Opus p = .007, Sonnet p = .07. Both give 3–4 distinct answers in 15 of 16 trunks, but with one sample per stimulus that can be partly noise. The firm evidence that answers track the readout is polarity given the pair (35/36 across both models), since f and NOT f share the first layer.
   - Opus picks the right input group above chance (42/64, per-trunk p = .025); Sonnet doesn't (33/64). Given the group, both get the pair above chance (15/33 and 21/42, vs about 10 and 11 expected). Given the pair, polarity is almost always right (14/15, 21/21).
   - Opus answered "none" on 15/64 rescaled readouts (7/32 at the old scale); Sonnet abstained once and truncated 5/64.
   - **Round 5:** under forced choice (no "none" option) Opus reaches 52% per trunk, 90% CI [39%, 64%], p = .004 against the 25% ceiling. **Opus decodes readouts when it can't abstain** (G3); see below.
   - Rescaling changed nothing. On the same 32 readouts, old scale vs rescaled: Sonnet 8 → 5, Opus 10 → 10. So the weight-size flaw didn't drive the result.
5. **Sonnet's XOR/parity answers partly follow the output-bias sign; Opus's don't** (exploratory). The pooled rates imply Sonnet follows the cue on about 26% of these stimuli and otherwise guesses. Round 4's new networks alone: 16/24 vs 9/24, p = .05. Pooled over all XOR/parity networks:
   - Sonnet: polarity right 34/54 when the bias sign points the right way vs 20/54 when it points the wrong way (27 vs 27 networks, permutation p = .005). Replies say so explicitly: "The bias is 1.01, which is positive, so the all-zero input is likely to give output 1."
   - Opus: 23/24 vs 38/42, p = .44.
   - The pre-registered replication in round 3b, on only 6 vs 6 networks, failed. The pooled result is significant, but it is exploratory, not confirmatory.
6. **Size may matter.** At 505 weights Sonnet got all 16 negation stimuli right, but the bias sign pointed the right way on all 16, so this doesn't separate reading from the shortcut.

**The account:** a zero-shot reader identifies the function from first-layer structure. Polarity is the output at the all-zero input, and the bias plus the 5–10 largest hidden-unit terms decide it on 94–100% of XOR/parity stimuli. Opus's accuracy is in that range; Sonnet's is no better than the bias sign alone. On shared trunks both extract some readout-specific information, but neither reliably identifies the function.

## Pre-registered verdicts
| Rule | Result | Verdict |
|---|---|---|
| D1, polarity on a clean break (Sonnet) | s = +75%, CI [+47%, +89%] | Polarity-sensitive |
| D2, readout decoding (Sonnet, 8 trunks, old scale) | 8/32, p = .69 | No evidence |
| D3, warning (Sonnet) | +17%, CI [−0%, +34%] | Inconclusive |
| E1, bias-cue replication (6 vs 6 networks) | 7/12 vs 6/12, p = .50 | Not replicated |
| E2, warning vs bias cue | +7%, CI [−13%, +26%] | Inconclusive |
| E4, Opus on trunks (old scale) | 10/32, p = .31 | No evidence |
| **F3, Opus vs Sonnet on XOR/parity (24 new networks)** | **44/48 vs 25/48, +40 points, p < .0001** | **Replicated** |
| **F4a, readout decoding at matched scale (16 trunks)** | Sonnet 22% [14%, 31%]; Opus 33% [22%, 44%], p = .20 | Inconclusive (Sonnet: no decoding; Opus: not excluded) |
| D5, Haiku | 62–100% truncated | Not usable above 505 weights |
| **G1, Sonnet on cue-reversed monotone networks** | 22/24 vs 46/46, drop 8 points, p = .11 | **Computes monotone polarity**; the shortcut is XOR/parity-specific |
| **G2, Opus on cue-reversed monotone networks** | 24/24 vs 34/34 | **Computes monotone polarity** |
| **G3, Opus on trunks under forced choice** | 52% [39%, 64%], p = .004 | **Opus decodes readouts under forced choice** |

Descriptive results:
- "None of the above" was given on 3 of 148 intact networks in round 3.
- Networks trained directly on NOT f were answered with a plain option only 1/8 times, so there is no prior against NOT answers.
- Opus at medium effort with an 8k cap truncated 10 of 16 answers. At low effort it truncated 2 of 162, both on rescaled trunks.

## Round 5: cue-reversed monotone networks, and forced choice on trunks
Pre-registered in `PLAN_5.md`, commit `e74e9f5`, pushed before the first call. 148 calls cost $14.23. Nothing was truncated or skipped, and all four steps finished. Reproduce with `python run_round5.py --report`.

**Stimuli.** MONO-CUE-214 has 12 new networks, 2 per monotone function. Each has b2 > 0 and more than half of W2 positive, which is the CUE-214 definition, and is shown as trained and negated, so the bias sign points the wrong way on all 24. Normal training rarely gives such networks: in 1,000 seeds per function, D gave 41, K gave 1, and A, B, I and J gave none. Per function, the first method that worked was used:
- D: normal training.
- B and K: training with b2 held at the NEG-214 mean |b2|.
- I: an exact shift through always-active units.
- A and J: the same shift, plus flipping the sign of 8–27 dead units' output weights. This is exact, and it was added before any call, by the author's decision.

One artifact is visible: the shift raises max |b1| (I 1.92 vs 1.22; A 1.67 vs 1.31). The full tables are in `PLAN_5.md`.

**G1, Sonnet at medium effort with a 6k cap: it computes monotone polarity** (rule 2).
- Polarity was right on 22/24 (92%), vs 46/46 on its cue-right monotone stimuli. That is a drop of 8 points; Fisher p = .11, and the network permutation test gives p = .11.
- On the originals, where the positive bias points to a NOT answer, it was right 12/12. On the negated networks it was right 10/12.
- Both errors name the original function on a negated network (`MC_A1_neg` → A, `MC_B1_neg` → B). That is the right pair with flipped polarity, the shortcut's signature, but here it can't be told apart from ignoring the output layer.
- On networks without the shift artifact (D, B and K): 11/12.
- Exact gate: 15/24. Pair: 17/24. 18 of 24 replies mention the output bias.

**G2, Opus at low effort with a 6k cap: it computes monotone polarity** (rule 2).
- Polarity 24/24, vs 34/34 on run 3's cue-right NEG-214 stimuli. Exact gate 19/24.
- Opus's first low-effort monotone run, on all 36 NEG-214 monotone stimuli: polarity 36/36, exact 25/36.

**Descriptive: the dead-unit flips mislead the gate, not the polarity.**
- On A and J (method d), exact accuracy was 1/8 for Sonnet and 3/8 for Opus. On the other four functions it was 14/16 and 16/16.
- All 5 of Opus's errors and 7 of Sonnet's 9 name a more permissive gate with the right polarity: AND → OR, 3-AND → MAJORITY, MAJORITY → 3-OR, or their NOTs. Six of Sonnet's seven are on A and J.
- A plausible reading: both models judge how permissive a gate is partly from the share of positive output weights, which the flips raise without changing the function. This is confounded with function, since only A and J were flipped.

**G3, Opus at low effort with a 6k cap, forced choice on the 16 rescaled trunks: it decodes readouts** (rule 1).
- Exact 33/64. Mean per-trunk accuracy 52%, 90% CI [39%, 64%]. Sign-flip test against the 25% ceiling: p = .004. Against 12.5% (exploratory): p = .0002.
- Levels: input group 50/64 (per-trunk p = .0002); pair given group 33/50, against about 13 expected from its answer frequencies; polarity given pair 33/33.
- Per-trunk correct counts: 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4. Every trunk got at least 3 distinct answers.
- No truncations and no "Q" answers. Mean output 2,295 tokens, vs 2,136 in round 4.
- Same 64 readouts, round 4 (Q allowed) vs forced choice: 21 → 33 right.
  - On the 15 readouts answered "none" in round 4: 7/15 right.
  - On the other 49: 21 → 26.

  So abstentions account for 7 of the 12-point gain. The rest is run-to-run variation, or an effect of the prompt change: the forced-choice prompt also implies that one option is right. With one sample per stimulus, the two can't be separated.
- F4a (round 4) stays inconclusive as pre-registered; G3 is a separate, forced-choice result. Sonnet was not run under forced choice.

**What this changes:**
- **Finding 2.** Both models compute monotone polarity even when the bias sign points the wrong way. Sonnet's bias-sign effect (exploratory) is specific to XOR and parity.
- **Finding 4.** Opus decodes shared-trunk readouts well above the first-layer ceiling when it has to pick a function, but about half its answers are still wrong. With "none" allowed it often abstains instead. Sonnet's trunk result (22%, with "none" allowed) is unchanged and untested under forced choice.

## Summary
- **Firm:** findings 2 and 3, and finding 1 as reframed above. Round 5 removes the bias-sign confound from finding 2.
- **With caveats:** finding 4 (Opus decodes under forced choice per round 5; with "none" allowed, inconclusive), finding 5 (exploratory) and finding 6 (confounded with the cue).
- **Limits:**
  - Claude only.
  - One sample per stimulus.
  - The Opus/Sonnet contrast compares low effort with medium effort; Sonnet given more output did not finish.
  - Rescaled trunks have smaller output biases than single-task networks (mean about 0.15 vs 1.43; the sign is unchanged).
  - The 16-option list lowered Sonnet's original-network accuracy from 79% to 71%, so compare within round 3 only.
  - MONO-CUE networks were built by a different method per function. A, I and J carry the shift artifact, and A and J also carry the dead-unit flips. Sonnet was not run on trunks under forced choice.
