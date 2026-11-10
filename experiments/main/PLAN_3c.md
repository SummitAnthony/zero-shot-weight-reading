# Round 3c: does Opus read the output layer when the first layer can't tell? (pre-registration)

Written Nov 5, 2026, after round 3b, before any round-3c call. In round 3b, Opus 5.5 at low effort never truncated (0/18). On the stimuli where Sonnet's polarity was right 5/18 times, Opus was right 17/18, with the exact gate 15/18. The open question that decides the headline is whether Opus also decodes readouts when the first layer is uninformative.

**Run** (`run_round3c.py`): Opus 5.5, low effort, 6,000-token cap, on TRUNK-214, trunk by trunk (T0 to T7), 2 parallel workers. That's 32 calls, expected about $3.30. The hard cap is $3.74, the user's $20 ceiling for round 3 minus the $16.26 already spent. If the cap cuts the run, only complete trunks are analysed.

**E4** (confirmatory). Take complete trunks only, and per-trunk exact accuracy minus 0.25 (the cap for a reader that uses only the first layer). An exact one-sided sign-flip test over trunks with p < .05 means "Opus reads the output layer when the first layer can't tell". Also reported, as in D2: the hierarchical breakdown (input group, pair, polarity) and Sonnet's results on the same stimuli.
