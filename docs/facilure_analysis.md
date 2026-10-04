# Failure investigation: attention after large spikes

Honesty note: I did not write predictions before running experiments/failure.py, so this analysis
is written after seeing the results. The failure was first seen in the shift experiment (spikes x2:
attention MAE 7.77 vs ridge 7.20, RMSE 14.94 vs 12.76). The diagnostic came afterwards.

## Scenario
Spikes x2 world, 10 test series (4080 windows). Target hours where a big spike (event effect > 15)
is still inside the 24 input hours: 681 windows, about 17%.

## Expected behaviour
Each spike lasts at most 6 hours, so 7 or more hours later it has no effect on the target. A good
model should ignore it: no change in the prediction when the spike is removed, and error close to
the no-event level (about 6.5).

## Actual behaviour
| hours since spike | n | attention MAE | attention bias | attention effect | ridge MAE | ridge effect | last value MAE |
|---|---|---|---|---|---|---|---|
| 4-6 | 78 | 12.72 | +8.39 | 11.81 | 9.31 | 11.40 | 7.24 |
| 7-12 | 156 | 15.15 | +11.53 | 14.27 | 7.98 | 8.89 | 7.61 |
| 13-24 | 301 | 13.57 | +6.88 | 10.39 | 11.03 | 7.74 | 7.99 |
| none in window | 3399 | 6.50 | -2.63 | 0.18 | 6.69 | 0.25 | 8.10 |
(bias = mean prediction minus truth; effect = how much the prediction changes when the events are
removed from the inputs.)
Attention's error is about twice its normal level and worse than last value. It over-predicts and
its prediction moves by 10-14 orders because of a spike whose true effect is zero. Ridge does the
same thing, but less.

## Explanation (a hypothesis, not proven)
The model carries the old spike value into its prediction. The attention weight on the spike's
position is below usual (0.64-0.75 times at lags 2-12), so attention is not locking onto the spike.
A likely cause, not tested: the output is a weighted sum of values, and the values grow with the
input size, while the weights are capped between 0 and 1. A small weight times a huge value still
moves the output. The spike inputs are also far outside the training range.

## Evidence
- Removing the events changes attention's prediction by 14.3 orders at lags 7-12, where the true
  effect is zero.
- Error and bias by hours since the spike, with no-event windows as the control.
- Attention weight on the spike position: 2.75 times usual at lag 1, 0.64-0.93 times at lags 2-24.

## Limits
Neighbouring windows share the same spike, so there are far fewer than 681 independent cases.
Small groups at lags 2-3 (27 and 26 windows). Weights come from one seed. The "effect" test removes
all events in the window, not only the big spike. Both models under-predict by about 3 in the
no-event group, which I did not investigate.

## Possible improvements (not tried)
Clip the normalised inputs (for example at 3 standard deviations), train with bigger spikes, or use
a robust loss such as Huber. A quick first test: clip the test inputs only and see if the effect and
bias shrink.