# Round 3b: replicate and probe the bias-sign shortcut (plan and pre-registration)

Written Nov 5, 2026, after round 3's results (`results_round3.jsonl`, $12.10) and before any round-3b API call. It is a follow-up motivated by those results, not part of round 3's pre-registration.

Budget: the user raised the ceiling for round 3 to $20 total, so round 3b's hard cap is $7.90. Expected spend about $4.30.

## What round 3 found that this round tests
In D1-cue, on XOR and parity networks, Sonnet's polarity was right 11/18 times when the output-bias sign pointed the right way, and 5/18 when it pointed the wrong way.
- An exact permutation test over the 9 vs 9 networks gives p = .038.
- Its replies reason from the bias directly ("The bias is 1.01, which is positive, so the all-zero input is likely to give output 1").

That finding was descriptive and borderline. Round 3b tests three things:
- E1: does it replicate on new networks?
- E2: is it a default strategy that the warning overrides?
- E3: does Opus at low effort finish, and does it share the shortcut?

## Stimuli and runs
`make_round3b.py` builds CUE2-214: 12 new XOR and parity networks at 214 units. Per function, 3 have the bias cue reversed and 3 have it as usual, each shown as trained and negated. That's 24 stimuli, built the same way as CUE-214, with seeds from 600000.

| Step | Run | Model, effort, cap | Stimuli | Calls |
|---|---|---|---|---|
| 1 | R1 (unwarned) | Sonnet 5.5, medium, 6,000 | CUE2-214 | 24 |
| 2 | R3 warned | Sonnet 5.5, medium, 6,000 | The cue-wrong XOR/parity stimuli in CUE-214 and CUE2-214 (NEG-214's 6 already have warned answers) | 24 |
| 3 | R4b | Opus 5.5, low, 8,000 | The 18 cue-wrong XOR/parity stimuli in NEG-214 and CUE-214 | 18 |

## Pre-registered rules
"Polarity right" and the Tango interval are as in PLAN.md.

- **E1, replication** (Sonnet, CUE2-214 only, 6 vs 6 networks). Count polarity-right answers per network, out of 2. *Replicated* if the cue-right rate minus the cue-wrong rate is ≥ 30 points and the exact one-sided permutation test over networks gives p < .05. Also reported: the pooled test over every XOR/parity network in rounds 3 and 3b.
- **E2, warning vs shortcut** (Sonnet, all 30 cue-wrong XOR/parity stimuli, warned vs unwarned, paired). This uses the gain in polarity-right answers.
  - Lower bound above 0 → "the warning overrides the shortcut: a default strategy".
  - Upper bound below 20 points → "the shortcut persists under the warning".
  - Otherwise inconclusive.
- **E3, Opus at low effort** (descriptive). Truncation rate, polarity right, how often a real answer follows the cue, and Sonnet on the same 18 stimuli.

## Files
- `make_round3b.py` builds `stimuli_round3b.jsonl`.
- `run_round3b.py` reuses `run_round3.py`'s call, guard and statistics.
- `test_round3b.py` holds 3 tests.
- `PREREGISTRATION_3b.txt` records the hashes and UTC time, written before the first round-3b call.
