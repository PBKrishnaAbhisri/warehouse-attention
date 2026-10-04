# From First Principles: Warehouse Demand Attention Model

Author: P.B.Krishna Abhisri
Demo video: [PASTE GOOGLE DRIVE LINK]

A small experiment with two connected parts:
1. **Scaled dot-product self-attention built from raw tensor operations** (no attention or Transformer
   library), verified with tests and a finite-difference gradient check, and shown learning a toy
   associative-recall task.
2. **A warehouse demand model** that uses that same attention code to predict the next hour of orders
   from the previous 24 hours of a synthetic warehouse, compared with simple baselines, a linear control,
   a distribution-shift test and a failure investigation.

## Main findings (details in docs/experiment_report.md)
| Question | Result |
|---|---|
| Does the attention code learn? | Yes: 100% test accuracy on associative recall (chance 10%) |
| Does dividing by sqrt(d_k) matter? | Yes when scores start large (unit init, d_k = 64: 225-350 steps to 90% vs 50, and 94.6% vs 100% accuracy). With the default small init, unscaled was slightly faster |
| Does attention beat simple rules on warehouse demand? | Yes: test MAE 5.98-6.14 vs 7.37 (last value), 12.94 (same hour yesterday), 15.81 (moving average) |
| Does attention beat a plain linear model (ridge)? | No: ridge 6.04 vs attention 5.98 on the official test, within the seed spread. On this additive data fixed weights are enough |
| Under noise x2? | Attention and ridge both degrade about 61-62% |
| Under spikes x2 (size and frequency)? | Attention degrades more (+26%, MAE 7.77) than ridge (+16%, MAE 7.20) |
| Failure | After a large spike, attention keeps reacting to it for up to 24 hours (prediction moves 10-14 orders, over-predicts), though the spike's true effect ends after 6 hours. Cause not confirmed |

Several of my own predictions were wrong. They are scored honestly in the docs.

## Repository layout