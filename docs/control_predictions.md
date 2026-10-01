# Control experiment predictions (written before running experiments/controls.py)

Honesty note:  I had seen the attention results (6.14 official test MAE) and the baselines, but not any of these controls.

1. Ridge regression (linear, 24x5 + target time features) will have test MAE above attention,
   roughly 6.5-8. Reason: it cannot form weekday x level interactions, so it should fail on
   Mon/Sat/Sun like same-hour-yesterday does.
2. Uniform-attention model (all weights 1/24, same head) will be worse than learned attention by
   more than 0.5 MAE. Reason: an equal-weight average discards which hour is which.
3. Training 200 epochs instead of 60 improves attention test MAE by only 0.1-0.5
   (best epochs 59/57 suggest it was still slowly improving).
4. What would change my mind: if ridge or uniform-attention is within 0.2 MAE of learned
   attention, then attention is not adding measurable value on this data, and I will say so.

   ## Results and scoring (written after running experiments/controls.py)

| Method | Official test MAE | Pooled MAE (4080 windows) |
|---|---|---|
| last value | 7.37 | 7.69 |
| ridge (linear, lambda=10) | 6.04 | 6.22 |
| attention, 60 epochs (3 seeds) | 6.14 | 6.32 |
| attention, 200 epochs (3 seeds) | 5.98 | 6.17 |
| uniform attention, 200 epochs | 8.21 | 8.04 |

1. WRONG. Ridge matched attention (6.04 vs 5.98). My reasoning (interactions needed) was wrong
   because the generator is additive, so a linear model can represent it.
2. RIGHT. Uniform attention is about 2 MAE worse.
3. RIGHT (low end). 200 epochs gives about 0.15 improvement, smaller than the seed spread (5.68-6.16).
4. TRIGGERED. Ridge is within 0.2 MAE of learned attention, so on this data attention adds no
   measurable value over a linear model.

Interpretation (hypotheses, not proven): the useful positions in this environment are fixed, so
fixed per-position weights (ridge) suffice; attention's content-based selection is not needed.
In the toy recall task the needed position changes per example, and there attention was essential.
Untested: ridge error by weekday; whether a generator with content-dependent structure would
separate the two.