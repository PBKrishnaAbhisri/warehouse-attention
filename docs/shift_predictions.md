# Distribution-shift predictions (written before running experiments/shift.py)

Honesty note: draft suggested by Claude, edited and adopted by me. I had already seen the three
fixed baselines under shift (from baselines_eval: last value 7.7 -> 12.6 with noise x2, 8.6 with
spikes x2). I had NOT seen ridge or attention under shift. Models are trained only on the original
distribution. H3 in docs/phase0_design.md is the committed hypothesis these refine.

1. Noise x2: attention and ridge overall MAE rise by 50-75% (from about 6.2 to roughly 9.5-11)
   and both stay below last value (12.6) in absolute terms. Reason: the irreducible noise doubles
   (floor about 4 -> 8), and models trained at sigma = 5 cannot change how much they average.
2. Spikes x2: MAE on normal hours changes by less than 10%; MAE at event hours rises 40-100%;
   overall MAE rises 8-20%. Reason: spikes are unpredictable from history, so bigger spikes mean
   bigger errors only at the hours they hit.
3. Attention vs ridge: the gap stays within 0.3 MAE in all three worlds. Reason: the generator is
   additive and the useful positions are fixed, so nothing separates them.
4. (Least certain) On windows containing a very large input, attention weights on the last row
   become sharper (lower entropy) than on normal windows, by more than 0.1. Reason: larger inputs
   mean larger Q.K scores, and the scores feed the softmax.
5. What would change my mind: if attention degrades by clearly less or more than ridge (more than
   0.5 MAE or 5 percentage points) in any world, then my "same structure, same behaviour" story
   is wrong.

## Results and scoring (written after running experiments/shift.py)

Pooled over seeds 1-10 (4080 windows). Models trained only on the original distribution.

| Method | control MAE | noise x2 MAE | spikes x2 MAE | spikes x2 RMSE |
|---|---|---|---|---|
| last value | 7.69 | 12.57 | 8.57 | 14.21 |
| ridge | 6.22 | 10.10 | 7.20 | 12.76 |
| attention (3 seeds, 200 ep) | 6.17 | 9.94 | 7.77 | 14.94 |

1. RIGHT. Attention +61%, ridge +62%, both below last value.
2. PARTLY WRONG. Ridge: normal +7%, events +29% (predicted 40-100), overall +16%. Attention: normal
   +16%, events +38%, overall +26% (predicted <10%, 40-100%, 8-20%).
3. WRONG in the spikes world: attention is 0.57 MAE worse than ridge (0.05 control, 0.16 noise).
4. DIRECTION RIGHT: entropy on big-input windows is lower in all worlds (0.09, 0.06, 0.27 lower);
   it exceeds 0.1 only in the spikes world.
5. TRIGGERED: spikes x2 gives a 0.57 MAE gap and 10 percentage points more degradation than
   ridge, so "same structure, same behaviour" is false under larger spikes.

Interpretation (hypotheses, not proven): under noise both models degrade alike (about 0.88 x sigma
noise part + a fixed part of about 3.8, a two-point fit assuming quadrature addition). Under larger
spikes, attention seems pulled toward very large input values (sharper weights) and its RMSE is
worse than last value's. Untested: per-seed spread under shift, error vs hours since the spike.
Caveats: event-hour sets differ between worlds (60 vs 122 hours); 'big input' also fires on trend
and noise outside the spikes world.