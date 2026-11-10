# Round 4: a targeted round to firm up findings 3 and 4 (plan and pre-registration)

Written Nov 5, 2026, after rounds 3, 3b and 3c ($19.37; $23.61 across all API work), before any round-4 call. The user chose this option: one targeted round of about $20, the last spend before the preprint. Hard cap $20.00 in `run_round4.py`. Expected about $17.50.

## Why
Two findings are the most interesting and the easiest to dismiss:
- **Finding 3.** Opus reads XOR/parity polarity that Sonnet misses (17/18 vs 5/18). It rests on 9 networks.
- **Finding 4.** Neither model decodes a readout when several functions share one first layer. It rests on 8 trunks whose weights were about 10× larger than the single-task networks'.

## Stimuli (`make_round4.py`, free; 112 stimuli, every label checked after rounding)
- **TRUNKR-214** (64 stimuli). 16 shared trunks × 4 readouts: round 3's T0–T7 (same seeds and readouts) and 8 new trunks T8–T15. All are rescaled to single-task weight statistics with two exact ReLU symmetries:
  - per unit, scale the input weights and bias by c and divide the output weight by c, with c set so the units' input norms follow the single-task distribution by rank;
  - per readout, scale all output weights and the output bias by k > 0, to match the single-task mean |W2|.

  Result: mean |W1| 0.17 vs 0.18 for single-task networks, and mean |W2| matched. Remaining mismatch: the output bias is smaller (mean about 0.15 vs 1.43). Its sign, the polarity cue, is unchanged.
- **XL2-214** (48 stimuli). 24 new single-task XOR and parity networks: 6 with the bias cue reversed and 6 as usual per function. Each is shown as trained and fully negated. Seeds start at 700000 (new trunks at 800000), both new.

## Runs, in priority order (`run_round4.py`)
| Step | Model, effort, cap | Stimuli | Calls |
|---|---|---|---|
| 1, 2 | Sonnet medium 6k, then Opus low 6k | Rescaled T0–T7 | 32 + 32 |
| 3, 4 | Sonnet medium 6k, then Opus low 6k | Rescaled T8–T15 | 32 + 32 |
| 5, 6 | Sonnet medium 6k, then Opus low 6k | XL2-214 | 48 + 48 |
| 7 | Sonnet **high** 12k | XL2-214, 8 networks (C0, C1, L0, L1; rev and norm; both versions) | 16 |

Steps 1–4 cost at most $18.82 even if every call hits its cap, so they always finish.

## Pre-registered rules
Definitions are as in PLAN.md. A truncated or missing answer counts as wrong. Sign-flip tests are exact.

**F4a, readout decoding at matched scale** (confirmatory). Per model, take each complete trunk's exact accuracy minus 0.25, the ceiling for a reader that uses only the first layer, and test with a one-sided sign-flip test over the 16 trunks.
- p < .05 for a model → "that model decodes readouts when the first layer can't tell".
- Neither p < .05, and both upper 90% bootstrap bounds on mean per-trunk accuracy below 40% → "confirmed: neither model decodes readouts".
- Otherwise inconclusive.
- Also reported: input group, pair given group, and polarity given pair.

**F4b, the effect of scale** (descriptive). Each model on the same 32 readouts of T0–T7, old scale vs rescaled.

**F3, Opus vs Sonnet on XOR/parity** (confirmatory). On the 24 new networks, count polarity-right answers per network (out of 2, original + negated) for each model. Test Opus minus Sonnet with a one-sided sign-flip test over networks. *Replicated* if the gap is ≥ 20 points and p < .05. Also reported: exact-gate accuracy and the split by bias-cue direction.

**F3-effort, more thinking for Sonnet** (descriptive). On 16 stimuli, Sonnet at high effort with a 12k cap vs Sonnet medium and Opus low on the same stimuli. It checks whether Sonnet closes the gap given more output; on the earlier stimuli Opus wrote about 30% more than Sonnet.

## Files
- `make_round4.py` builds `stimuli_round4.jsonl`.
- `run_round4.py` reuses `run_round3.py`.
- `test_round4.py` holds 7 tests (`python -m pytest test_round4.py test_round3.py test_round3b.py -q`).
- `PREREGISTRATION_4.txt` records the hashes and UTC time.
