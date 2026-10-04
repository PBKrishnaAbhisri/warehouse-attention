# Experiment report

Environment: Python 3.10.0, torch 2.14.1 (CPU), numpy 2.2.6, matplotlib 3.10.9, pytest 9.1.1.
Seeds: data seed 0 for training; seeds 1-10 for the shifted and pooled test sets; model seeds 0, 1, 2.
Run commands (from the project root): `python -m pytest -v`, then `python -m experiments.gradient_check`,
`toy_task`, `ablation`, `baselines_eval`, `warehouse`, `controls`, `shift`, `failure`.
Process honesty: Phase 0 was committed before any experiment. The ablation predictions were committed
together with its results. The model predictions (docs/model_predictions.md) were written after the run,
and no predictions were written before the failure investigation. Those parts are labelled post-hoc.

## 1. Verification
- 8 attention tests passed (shapes, softmax rows sum to 1, stable softmax, match with PyTorch's built-in
  attention used only inside a test, large logits, gradcheck). 6 data tests were added (determinism,
  noise level, window boundaries).
- Gradient check (float64, eps 1e-6): W_Q analytic 1.58320963 vs numeric 1.58320963 (relative difference
  1.7e-10); W_K 1.28788688 vs 1.28788688 (2.7e-10); W_V 5.04160232 vs 5.04160232 (2.8e-11).
- Numerical stability: naive softmax on [1000, 1001, 999] gives NaN; the stable version is valid.

## 2. Toy task: associative recall
8 key-value pairs (16 keys, 10 values), then a query key. d_k = 16, Adam lr 0.01, batch 64, 1500 steps.
Loss 2.317 (= chance, ln 10) fell to 0.0001; batch accuracy went from 0.08 to 1.00 by step 150; accuracy on
2000 fresh puzzles was 1.000 (chance 0.100). In one example the query row gave 0.506 weight to the
matching pair and 0.452 to itself (same key). Entropy went from 2.196 (uniform) to 0.873. Late in training
the gradient norm of W_Q and W_K fell (0.0075 to 0.0001) simply because the loss was near zero; this is
different from the saturation effect below.

## 3. Ablation: scaled vs unscaled (steps to reach 90% accuracy, 3 seeds)
| init | d_k | scaled | unscaled | start entropy (scaled / unscaled) |
|---|---|---|---|---|
| default (std 1/sqrt(d_in)) | 4 | 150, 175, 175 | 125, 175, 150 | 2.196 / 2.192 |
| default | 16 | 100, 75, 75 | 75, 50, 75 | 2.196 / 2.178 |
| default | 64 | 50, 50, 50 | 25, 25, 25 | 2.196 / 2.126 |
| unit (std 1.0) | 4 | 275, 275, 300 | 300, 300, 275 | 1.655 / 1.117 |
| unit | 16 | 125, 100, 100 | 250, 225, 250 | 1.605 / 0.497 |
| unit | 64 | 50, 50, 50 | 225, 350, 350 | 1.651 / 0.265 |
All test accuracies were 1.000 except unscaled, unit init, d_k = 64: 0.946 (final entropy 0.027).
Hypothesis H1 (Phase 0) assumed unit-variance Q and K. That does not hold for the default init, so
I ran two conditions (written down before running). Result: with unit init the gap grows with d_k
(about 1x, 2x, 4.5-7x), matching the prediction; with the default init unscaled was slightly faster.
Explanation: large scores make the softmax almost one-hot, where its derivative is near 0.

## 4. Data and baselines
Demand = base 100 + trend 0.01 t + daily bumps (11:00, 16:00) + weekend dip + spikes/drops + Gaussian
noise (sigma 5). Split by time: days 1-84 train, 85-102 validation, 103-120 test. Official test MAE
(RMSE): last value 7.37 (9.99); 24h moving average 15.81 (18.29); same hour yesterday 12.94 (17.18).
Phase 0 expected same-hour-yesterday at 6-9; it is 5.7-7.0 on Tue-Fri but 25.0 on Monday, 15.6 on
Saturday, 10.4 on Sunday, because yesterday was a different weekly level.

## 5. Warehouse attention model and controls
One attention layer (d_k = 16), a linear head on the last hour's summary plus the target hour's time
features, MSE loss, Adam lr 0.003, batch 64, best epoch by validation MAE.
| MAE (RMSE) | official test (408 windows) | pooled control (4080 windows) |
|---|---|---|
| last value | 7.37 (9.99) | 7.69 (10.51) |
| ridge (linear) | 6.04 (8.43) | 6.22 (8.81) |
| attention, 60 epochs | 6.14 (8.64) | 6.32 (9.07) |
| attention, 200 epochs | 5.98 (8.60) | 6.17 (8.92) |
| uniform attention, 200 epochs | 8.21 (11.42) | 8.04 (11.22) |
Attention seeds on the official test: 60 epochs 6.19, 6.05, 6.19; 200 epochs 6.09, 5.68, 6.16.
Finding: attention beats the fixed rules, but ridge matches it (within 0.06), so on this data attention
adds no measurable value over a linear model. Uniform attention is about 2 MAE worse, so the learned
weights per position matter. The generator is additive and the useful positions are fixed, so fixed
weights are enough. The attention weights (Tue-Fri vs Mon/Sat/Sun) are nearly identical and flat
(max 0.09 against uniform 0.042), so the plot does not show a weekday-dependent choice. Weight plots
are a visualisation, not a causal claim.

## 6. Distribution shift (trained once on the original data; pooled over seeds 1-10)
| MAE | control | noise x2 (sigma 10) | spikes x2 (size, frequency) |
|---|---|---|---|
| last value | 7.69 | 12.57 | 8.57 |
| same hour yesterday | 12.26 | 15.75 | 14.02 |
| ridge | 6.22 | 10.10 | 7.20 |
| attention | 6.17 | 9.94 | 7.77 |
Noise x2: attention +61%, ridge +62%. Spikes x2: attention +26% (RMSE 14.94), ridge +16% (RMSE 12.76).
Under larger spikes attention is 0.57 MAE worse than ridge, and its RMSE is worse than last value's (14.21).
Predictions that held: noise degradation. Predictions that failed: attention close to ridge under spikes.

## 7. Failure investigation
After a big spike, 4-24 hours later, attention over-predicts (bias +7 to +12) and its prediction moves
by 10-14 orders when the events are removed from the inputs, although the spike's true effect is zero.
Ridge does this less. The attention weight on the spike's position is below usual, so the cause is not
attention locking onto the spike. Untested explanation: values grow with the input size while weights
are capped at 1. Details and limits: docs/failure_analysis.md.

## 8. Limits
Synthetic data with an additive structure; one-layer model; 3 model seeds; only 6 event hours in the
official test split (pooled sets used for events); event-hour comparisons are noisy; the failure
analysis used one seed for weights; the failure cause is a hypothesis.