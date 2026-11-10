# Round 3 pilot: does Claude read the output layer? (plan and pre-registration)

Written Nov 5, 2026, before any round-3 API call. Revised the same day after two independent reviews (a hostile methodologist and a statistician, both offline). Budget: hard cap $13.00 in `run_round3.py` (the user authorized $10–15). Expected spend about $10.60.

## Why this round exists
The earlier "broken" networks flipped the output sign of a random subset of hidden units. Most of them turned out to be dead networks: of the 24 broken ~3,000-weight networks, 14 output a constant or near-constant value, and 8 are exactly constant over all 4,096 inputs. Their only correct answer was "none of the above". Stage 1's flips (`08_Stage1_Size_Curve`) have the same problem: 28 of 64 are constant or near-constant, and the share differs by size.

Round 3 replaces them with clean breaks that have a real, named answer, plus controls for the shortcuts the reviewers found.

## Stimuli (`make_round3.py`, free; 136 stimuli, every label checked on the displayed weights)
**Options:** 16 functions plus Q ("none"), closed under negation:
- On x2 and x5: AND, OR, XOR, x2 AND NOT x5, and the NOT of each.
- On x0, x3 and x7: MAJORITY, AND, OR, PARITY, and the NOT of each.

Every plain option outputs 0 on the all-zero input; every NOT option outputs 1. No stimulus's answer is Q.

| Set | What it is | Stimuli |
|---|---|---|
| NEG-214 | The 24 single-task ~3,000-weight networks already used (8 from Pilot 2, 16 from round 2), as trained (f) and fully negated (every output weight and the output bias change sign: exactly NOT f, every magnitude kept) | 48 |
| CUE-214 | 12 new XOR and parity networks: 6 whose output bias and majority of output weights point the "wrong" way (positive, for a function with f(0) = 0) and 6 the usual way, each as trained and negated | 24 |
| NATIVE-214 | 8 networks trained directly on each NOT option, as trained and negated (the negation computes the plain option) | 16 |
| TRUNK-214 | 8 shared trunks (214 ReLU units, trained on all 16 options at once). Each shows 4 readouts, one complementary pair per input group, so the 4 stimuli share an identical first layer. Every pair appears in exactly two trunks; T0–T3 cover each pair once. | 32 |
| NEG-36 | The 8 single-task 505-weight networks from Pilot 2, as trained and negated | 16 |

New seeds start at 300000, outside every earlier range (Stage 1 used 70000–101749).

**Known cues, from the reviews:**
- **Bias sign.** In NEG-214 the output bias predicts polarity (bias below 0 means a plain option) for 40 of 48 stimuli. CUE-214 reverses that cue on 12 stimuli.
- **Symmetry.** Training is sign-symmetric, so a negated f-network is distributed exactly like a network trained on NOT f; the first layer carries no polarity information. NATIVE-214 checks this directly.
- **Trunk scale.** Trunk weights are about 10× larger than single-task weights (mean |W1| 1.74 vs 0.18), so trunk accuracy is not compared with single-task accuracy except descriptively.

## Runs (`run_round3.py`), in strict priority order
| Step | Run | Model, effort, output cap | Stimuli | Calls |
|---|---|---|---|---|
| 1 | R1 | Sonnet 5.5, medium, 6,000 | NEG-214 | 48 |
| 2 | R1 | Sonnet 5.5, medium, 6,000 | TRUNK-214 | 32 |
| 3–4 | R2 | Haiku 5.5, low and medium, 6,000 | Pilot 2's 8 NEG-214 pairs | 32 |
| 5 | R3, warned | Sonnet 5.5, medium, 6,000 | NEG-214, originals and negated | 48 |
| 6 | R1 | Sonnet 5.5, medium, 6,000 | CUE-214 | 24 |
| 7 | R1 | Sonnet 5.5, medium, 6,000 | NATIVE-214 | 16 |
| 8 | R3, warned | Sonnet 5.5, medium, 6,000 | TRUNK-214, trunks T0–T3 | 16 |
| 9 | R4 | Opus 5.5, medium, 8,000 | Pilot 2's 8 NEG-214 pairs | 16 |
| 10 | R1 | Sonnet 5.5, medium, 6,000 | NEG-36 | 16 |

Steps 1–5 cost at most $12.52 even if every call hits its cap, so they always finish. If spend runs high, the guard cuts from step 10 backward.

Warning sentence (R3 only): "Caution: some networks were edited after training, for example in their output layer, so the first layer alone may not tell you the function. Check how the output layer combines the hidden units before you answer."

## Pre-registered decision rules
**Common to all rules:**
- One sample per stimulus. A truncated or missing answer counts as wrong.
- Intervals are 90% Tango score intervals for paired differences.
- D1, D2 and D3 are confirmatory and tested in that fixed order at α = .05. The rest are descriptive.

**Definitions:**
- *Polarity sensitivity*, on pairs of the same network: s = P(plain answer | original) − P(plain answer | negated), with b = pairs where the original got a plain answer and the negated version a NOT answer, and c = the reverse.
  - A reader blind to the output layer has s ≈ 0, whatever its preference for plain names.
  - A reader that uses the output layer has s well above 0.
- *Polarity right*: the answer is one of the 16 options, and it is plain exactly when the label is plain.

**D1, polarity on a clean break** (Sonnet R1, NEG-214, 24 pairs). Uses s.
- Lower bound above 0 → "polarity-sensitive: uses the output layer for polarity".
- Upper bound below 0.30 → "blind to the output layer for polarity".
- Otherwise inconclusive.
- Also reported: originals and negated accuracy, both versions right, and negated named as f.

**D1-cue** (descriptive; interpretable only if D1 is sensitive). On every XOR and parity original/negated stimulus (CUE-214 plus NEG-214), compare the polarity-right rate when the bias cue points the right way with when it points the wrong way.
- A gap of ≥ 30 points with one-sided Fisher p < .05 → "bias-sign shortcut".

**D1-native** (descriptive). Accuracy and plain-answer rate on networks trained on NOT f. If they are answered plain about as often as the negated networks, the difficulty comes from a prior against NOT answers, not from the edit.

**D2, the output layer when the first layer can't tell** (Sonnet R1, TRUNK-214, 8 trunks × 4). A reader that uses only the first layer gives one answer per trunk, so it is capped at 25% per trunk.
- Exact one-sided sign-flip test over trunks on (per-trunk accuracy − 0.25), p < .05 → "reads the output layer when the first layer can't tell".
- Also reported, by level: input group right (chance 1/2), pair right given group (1/4), polarity right given pair (1/2), and distinct answers per trunk.
- Originals vs trunk accuracy is descriptive only, because of the scale confound.

**D3, default strategy or capability limit** (Sonnet, NEG-214; R3 warned vs R1 unwarned, the same stimuli). The gain is on the 24 negated stimuli, paired.
- If warned originals drop more than 10 points → "shift toward NOT answers, not reading".
- Otherwise, lower bound above 0 → "default strategy: the ability is there".
- Otherwise, upper bound below 20 points → "no large gain".
- Otherwise inconclusive.
- Polarity sensitivity under the warning and the warned trunk results are descriptive.

**D4, Opus** (descriptive). Polarity metrics on Pilot 2's 8 pairs.

**D5, Haiku.** A setting is usable only if it truncates ≤ 10% of answers and gets ≥ 3 of 8 originals right. Otherwise drop Haiku above 505 weights.

**D6, 505 weights** (descriptive). Polarity metrics on NEG-36. The bias cue holds for all 8 of its originals, so a smaller gap there is expected even from a bias-sign reader.

**Also logged:**
- the plain-option share of trunk answers (true share 50%, a free check of a plain-name prior);
- replies that mention the output bias or the all-zero input;
- "none" (Q) answers on intact networks;
- output tokens per run, for the main-study budget.

## Files
- `make_round3.py` builds `stimuli_round3.jsonl`.
- `test_round3.py` holds 14 tests on the stimuli, statistics and call plan (`python -m pytest test_round3.py -q`).
- `run_round3.py` runs the calls (`--dry-run`, `--report`).
- `PREREGISTRATION.txt` records the SHA-256 hashes and the UTC time, written before the first paid call.
