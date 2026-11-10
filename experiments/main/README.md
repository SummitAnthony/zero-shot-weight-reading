# Rounds 3–5: the main experiment

This folder holds the whole main experiment, rounds 3, 3b, 3c, 4 and 5. In the research repository this folder was `09_Round3_Readout_Pilot`, and the plans, run logs and code comments use the original folder names: `04_Pilot2_Large_Networks` is now `../pilot/networks`, `07_API_Pilot` is `../pilot` and `10_Paper` is `../../paper`. For each round there is a plan, a pre-registration record, a stimulus generator, the stimuli, a runner with a hard budget cap, tests, every reply, and a run log.

[`RESULTS.md`](RESULTS.md) is the write-up made round by round as the results came in. The paper is the definitive account, and [`../../paper/analysis.py`](../../paper/analysis.py) computes every number in it from the files here.

## Files by round
| Round | Plan | Pre-registration | Generator | Stimuli | Runner | Tests | Replies | Log |
|---|---|---|---|---|---|---|---|---|
| 3 | `PLAN.md` | `PREREGISTRATION.txt` | `make_round3.py` | `stimuli_round3.jsonl` | `run_round3.py` | `test_round3.py` | `results_round3.jsonl` | `run_log.txt` |
| 3b | `PLAN_3b.md` | `PREREGISTRATION_3b.txt` | `make_round3b.py` | `stimuli_round3b.jsonl` | `run_round3b.py` | `test_round3b.py` | `results_round3b.jsonl` | `run_log_3b.txt` |
| 3c | `PLAN_3c.md` | `PREREGISTRATION_3c.txt` | (round 3's) | (round 3's trunks) | `run_round3c.py` | (round 3's) | `results_round3c.jsonl` | `run_log_3c.txt` |
| 4 | `PLAN_4.md` | `PREREGISTRATION_4.txt` | `make_round4.py` | `stimuli_round4.jsonl` | `run_round4.py` | `test_round4.py` | `results_round4.jsonl` | `run_log_4.txt` |
| 5 | `PLAN_5.md` | `PREREGISTRATION_5.txt` | `make_round5.py` | `stimuli_round5.jsonl` | `run_round5.py` | `test_round5.py` | `results_round5.jsonl` | `run_log_5.txt` |

**Do not edit the hash-locked files.** The plans, generators, runners, tests and stimuli listed in a `PREREGISTRATION*.txt`, and the earlier replies listed there, are fixed by their SHA-256 hashes. To check them, run this from this folder:

```bash
python check_preregistration.py
```

Every hash-locked file is byte-identical to its pre-registered version except `make_round3.py`. There, two lines that locate the pilot's networks were updated for this layout. The script puts the original two lines back in memory and confirms the original hash, without changing the file.

## Stimulus sets
Every set except `NEG-36` uses 214-unit networks (2,997 weights). "Trained and negated" means each network is shown once as trained and once fully negated.

| Set | Round | Stimuli | Contents | Main use in the paper |
|---|---|---:|---|---|
| `NEG-36` | 3 | 16 | The pilot's 8 networks at 505 weights, trained and negated | §4.2 |
| `NEG-214` | 3 | 48 | The pilot's 24 networks at 2,997 weights, trained and negated | §4.2–4.4 |
| `NATIVE-214` | 3 | 16 | 8 networks trained directly on NOT *f*, trained and negated | §4.2 |
| `CUE-214` | 3 | 24 | 12 XOR/parity networks with the bias cue reversed, trained and negated | §4.3, 4.4 |
| `TRUNK-214` | 3 | 32 | 8 shared trunks × 4 readouts, at the original scale | §4.5 |
| `CUE2-214` | 3b | 24 | 12 new cue-reversed XOR/parity networks (the bias-cue replication), trained and negated | §4.4 |
| `TRUNKR-214` | 4 | 64 | 16 shared trunks × 4 readouts, rescaled to single-task weight statistics | §4.5 |
| `XL2-214` | 4 | 48 | 24 new XOR/parity networks, trained and negated (the Opus vs. Sonnet comparison) | §4.3, 4.4 |
| `MONO-CUE-214` | 5 | 24 | 12 monotone-gate networks with the bias cue reversed, trained and negated | §4.2 |
| `NEG-214` | 5 | 36 | The monotone-gate part of round 3's `NEG-214`, shown to Opus | §4.2 |
| `TRUNKR-214-NOQ` | 5 | 64 | Round 4's rescaled trunk readouts, without the "none of the above" option | §4.5 |

## Reports and tests
No API key is needed for these:

```bash
python run_round5.py --report     # likewise run_round3.py, run_round3b.py, run_round3c.py, run_round4.py
python -m pytest test_round3.py test_round3b.py test_round4.py test_round5.py -q
```

`--dry-run` prints a runner's plan and worst-case cost without calling the API. New calls need `ANTHROPIC_API_KEY` and cost money.
