# API pilot: the partial sign flip (go/no-go gate)

Runs the 32 Pilot 2 networks (505 and 2,997 weights; originals and output-sign flips) through the Claude API with capped output, then prints a go/no-go verdict.

> **Role in the paper.** This is the pilot reported in Section 4.1: 184 calls (72 to Sonnet 5.5, 112 to Haiku 5.5) for $4.24. All replies are in `results.jsonl`. It produced an apparent dissociation: originals were named, but networks "broken" by the partial sign flip were mostly missed. The paper shows that this does not mean the output layer is ignored. Most flipped networks are constant or near-constant, but Sonnet missed them as often when they were not: it caught 4/14 constant or near-constant networks and 3/10 others. On 19/24 networks the edit leaves the output on the all-zero input, and so the polarity, unchanged; it also keeps the output bias's sign and the balance of output-weight signs. From round 3 on ([`../main`](../main)), exact edits replaced the partial flip. The design and the go/no-go gate below are kept as the record of the pilot.

## Run it
```
cd experiments/pilot
pip install anthropic
```
Set your key. On Windows PowerShell: `$env:ANTHROPIC_API_KEY="sk-ant-..."`. On macOS/Linux: `export ANTHROPIC_API_KEY=sk-ant-...`
```
python pilot_api.py --dry-run      # free: shows the plan and worst-case cost
python pilot_api.py --count-only   # free: real tokens per weight via count_tokens
python pilot_api.py                # the pilot itself (rerun to resume if interrupted)
python pilot_api.py --report       # reprint the analysis
```

## What it runs
| Run | Model | Effort | Output cap | Calls | Worst case |
|---|---|---|---|---|---|
| sonnet_medium_6k | Sonnet 5.5 | medium | 6,000 | 32 | $2.63 |
| haiku_high_6k_x2 | Haiku 5.5 | high | 6,000 | 64 (2 samples each) | $0.26 |
| sonnet_high_16k_big | Sonnet 5.5 | high | 16,000 | 8 large networks | $1.57 |

Total worst case $4.46. Actual spend will be lower because most replies stop before the cap.

## Safety
- `BUDGET_USD = 5.00` at the top of the script is a hard cap. Before each call it reserves the worst-case cost (input plus the full output cap) and skips the call if that could exceed the cap.
- Also set a $5 spend limit in the Claude Console as a second safety net.
- Results are appended to `results.jsonl` after every call, so nothing is lost if it stops.

## The gate
The report checks, for Sonnet on ~3,000-weight networks:
1. Originals correct at 60% or more.
2. Broken networks caught clearly less often (a gap of about 30 points).
3. Mean output around 5,000 tokens or less, with few truncated answers.

All three pass: commit to the full study. Item 1 fails: the effect doesn't survive capped thinking, so drop or rescope. Item 3 fails: use the fallback budget in the plan.

## Round 2 (after round 1 came back borderline)
Round 1 passed checks 1 and 3, but check 2 rested on only 8 networks (4 of 8 broken caught, a 25-point gap). Round 2 adds 16 new ~3,000-weight networks, each shown as original and as broken, at the same Sonnet settings, plus a no-thinking Haiku run as a "glance" baseline.

```
python pilot_api.py --dry-run --round2-only
python pilot_api.py --round2-only
```
About $1.50 more (64 calls, worst case about $3.20). `BUDGET_USD` is now $6.00 total across both rounds, so the worst-case reservations fit.

**Decision rule (written Nov 4, 2026, before round 2 ran, but committed only afterwards; see the paper's Section 3.3).** Pool all 24 paired networks (8 from round 1, 16 new). GO only if all three hold:
1. Originals correct at 60% or more.
2. Originals correct minus broken caught is at least 25 points.
3. One-sided exact McNemar p < 0.05 on the pairs.

If check 1 fails, it's a no-go. If check 1 passes but 2 or 3 fails, reading works but there's no clear dissociation: rescope the headline. `make_extra.py` regenerates the round-2 networks (they're already in `extra/`).

Uses adaptive thinking with the `effort` setting, since Claude 5.5 models reject fixed thinking budgets. Prices are standard (non-batch), checked Nov 4, 2026.

## Folder contents

### `networks/`: the Pilot 2 networks
These are the single-task networks behind the pilot. There are 16 MLPs with 12 binary inputs and one ReLU hidden layer, of 36 units (505 weights) or 214 units (2,997 weights), each trained to 100% on one of 8 boolean functions.

The API pilot shows them as trained and with the partial sign flip. Round 3 reuses the originals, as trained and fully negated. `NEG-36` uses the 8 networks at 505 weights. `NEG-214` uses the 8 at 2,997 weights and the 16 in `extra/`. The `src` field in [`../main/stimuli_round3.jsonl`](../main/stimuli_round3.jsonl) names each source network.

| File | Contents |
|---|---|
| `make.py` | Trains the networks and writes the prompts and `answer_key.json` |
| `prompt_H36_set*.txt`, `prompt_H214_set*.txt` | The blinded prompt sets, with a 9-option list (eight plain functions and "none of the above") |
| `answer_key.json` | The labels |

An informal check made before the API pilot is not part of the paper and is not included.

### `extra/`: the round-2 networks
The 16 round-2 networks at 2,997 weights, regenerated by `make_extra.py`.

### `flip_followup/`: superseded, never run
A planned size-curve stage. Networks of 155 to 2,997 weights were to be shown as originals, with permuted weights, and with the partial sign flip. It was superseded and never run. In 28 of its 64 flip networks the output is constant or near-constant, so the flip condition doesn't test what it was meant to. Rounds 3–5 in [`../main`](../main) replaced it with exact edits.

The paper uses this folder for one descriptive number only: the 28/64 above. It appears in Section 4.1 as "a prepared follow-up set with the same procedure has 28/64", and [`../../paper/analysis.py`](../../paper/analysis.py) computes it from `stimuli_stage1.jsonl`. No API calls were made with these stimuli. Only the generator (`make_stage1.py`) and the stimuli (`stimuli_stage1.jsonl`) are included.
