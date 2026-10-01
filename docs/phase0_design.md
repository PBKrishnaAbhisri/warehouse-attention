# Phase 0: Design Before Code

## 1. Mathematical formulation
Task: predict the next hour of demand from the previous 24 hours.
- Input window: X ∈ R^(24 × d_in), one row per hour t-23 ... t.
- Each row holds [normalized demand, sin(hour), cos(hour), sin(day-of-week), cos(day-of-week)], so d_in = 5.
- Self-attention is permutation-equivariant (it ignores order), so the time features are what tell the model where each hour sits in the day and week.
- Attention: Q = XW_Q, K = XW_K, V = XW_V; Y = softmax(QK^T / sqrt(d_k)) V, with Y ∈ R^(24 × d_k).
- Context = last row of Y (the representation of hour t after it has attended to all 24 hours).
- Prediction head: ŷ(t+1) = w^T [context ; time features of hour t+1] + b. The head gets the target hour's time features because the model must know which hour it is predicting.
- Loss: MSE on normalized demand. Parameters: W_Q, W_K, W_V, w, b.

## 2. Synthetic dataset design
Demand(t) = Base + Trend(t) + Daily(h) + Weekly(dow) + Event(t) + Noise(t), clipped at 0.
- Length: 120 days = 2880 hourly points.
- Base = 100. Trend = 0.01 * t (slow growth, about +29 over the whole series).
- Daily(h): two Gaussian bumps, +40 centered at 11:00 and +30 centered at 16:00, both with width 2 hours (morning and afternoon order peaks).
- Weekly(dow): additive offsets Mon-Fri 0, Sat -15, Sun -25 (weekend dip).
- Noise: Gaussian, mean 0, sigma = 5, constant variance (homoscedastic) in training.
- Spikes: each hour starts a spike with probability 0.004 (about one per 10 days). Size is Uniform(40, 80), decaying exponentially over about 3 hours.
- Drops: each hour starts a drop with probability 0.002. Size is Uniform(-60, -30), lasting 2 hours.
- Reproducibility: everything is generated from a seeded NumPy generator (train seed 0, shift-test seed 1).

## 3. Statistical assumptions
- Noise is independent across hours (no autocorrelation).
- Daily and weekly patterns are deterministic and stable, so the same hour yesterday is informative and so are the last few hours.
- Events are rare and unpredictable from history, but spikes decay, so the hours right after a spike are partly predictable.
- Irreducible error: with sigma = 5, a perfect model that knows the seasonal pattern still has MAE ≈ 0.8 * 5 = 4 on non-event hours. No model should beat this floor.

## 4. Evaluation methodology
- Chronological split, no shuffling (shuffling would leak the future): days 1-84 train, 85-102 validation, 103-120 test. Windows never cross split boundaries.
- Normalization statistics (mean, std) come from the training set only.
- Primary metric: MAE (in orders per hour, interpretable, robust to spikes). Secondary: RMSE (punishes large spike errors). Also reported: MAE on spike hours only.
- All baselines are evaluated on the same test windows.

## 5. Baselines and expected behavior
1. Last value: ŷ(t+1) = y(t). Expected MAE ≈ 8-15: it ignores the daily shape and has noise error std ≈ sigma * sqrt(2) ≈ 7.
2. 24-hour moving average. Expected to be the worst: it smooths out the peaks, so it is biased at peak hours (MAE ≈ 15-25).
3. Same hour yesterday: ŷ(t+1) = y(t-23). Expected MAE ≈ 6-9 (noise from two points, about 5.6 plus small trend and weekend-boundary errors). This is the strongest simple baseline.

## 6. Hypotheses (written BEFORE any experiment is run)
- H1 (ablation): With d_k = 64 and unit-variance Q and K entries, q·k has standard deviation about sqrt(d_k) = 8, so unscaled softmax will be nearly one-hot. Prediction: unscaled attention shows much lower attention entropy and smaller gradients on W_Q and W_K, and reaches a higher final loss or trains slower than scaled. At d_k = 4 the difference will be small. Falsified if unscaled matches scaled at d_k = 64.
- H2 (baseline): The attention model beats last value and moving average on test MAE by more than 20%, but will NOT clearly beat same-hour-yesterday (within ±10%), because the data is mostly seasonal. Falsified if attention beats same-hour-yesterday by more than 15% or loses to last value.
- H3 (distribution shift): With noise sigma 5 -> 10, spike size x2 and spike probability x2, all models degrade. Error on non-event hours rises roughly in proportion to sigma, and the biggest errors will be at spike hours. Falsified if the attention model degrades by less than the baselines.

## 7. System-specific failure modes
- F1: Spike onset hours. A spike starts without warning, so every model will under-predict at the spike hour. Then last value may beat attention for 1-3 hours after the spike.
- F2: Near-uniform attention. If the weights stay about 1/24 everywhere, the model degenerates into a moving average and loses the peak shape.
- F3: Shift in input scale. Under larger spikes, inputs fall outside the normalized training range, so the model may extrapolate badly.
- F4: Weekend boundaries. Windows that span Friday to Saturday contain two different weekly levels, so "same hour yesterday" will be wrong there.

## 8. Planned generalization experiment
Train on the original distribution. Test on (a) the original test set, (b) noise sigma 10, (c) spike size x2 and spike probability 0.008. Report MAE for the attention model and all baselines under each setting, and compare with H3.

## 9. Toy task for Problem 1 (separate from the warehouse)
Associative recall: given a sequence of key-value pairs and a query key, output the matching value. This needs attention to focus on the right position, so it genuinely tests the mechanism.