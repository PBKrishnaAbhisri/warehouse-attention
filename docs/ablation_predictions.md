# Ablation predictions (written before running experiments/ablation.py)

Note: Phase 0 H1 assumed unit-variance Q and K. On the toy task with the default init
(std = 1/sqrt(d_in), one-hot inputs) the initial score std is only about 0.1-0.8, so that assumption
does not hold there. Condition B (init std = 1.0) restores it.

Setup: associative recall, d_k in {4, 16, 64}, scaled vs unscaled, 3 seeds, Adam lr 0.01,
1500 steps. Metrics: test accuracy at the end, first logged step where batch accuracy >= 0.9,
attention entropy at step 0 and at the end.

- Condition A (std = 1/sqrt(d_in)): little difference between scaled and unscaled for
  d_k <= 64, both near-uniform attention at the start. Unscaled may even learn slightly faster
  because scaling makes already-small scores smaller. Falsified if unscaled is clearly worse.
- Condition B (std = 1.0), d_k = 64: unscaled starts with near one-hot attention (entropy about 0.3)
  and learns more slowly or fails on some seeds (test accuracy at least 10 points below scaled,
  or steps-to-90% at least 2x scaled). Scaled reaches >= 99% test accuracy.
  Falsified if unscaled matches scaled.
- The gap should grow with d_k: small at d_k = 4, largest at d_k = 64.
- Caveat I'm unsure about: Adam rescales each parameter's step, so it may partly hide
  vanishing gradients. If unscaled still trains fine, that is a result I'd have to explain.