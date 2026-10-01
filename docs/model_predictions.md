# Warehouse model predictions (written before running experiments/warehouse.py)

Honesty note: I had already seen the baseline results (last value 7.4, 24h moving average 15.8,
same-hour-yesterday 12.9 on the official test split) before writing these. The attention model
itself had not been trained.

Model: 1 attention layer (d_k = 16), linear head on the last row's summary plus the target hour's
time features, MSE loss, Adam, best epoch chosen on validation MAE.

1. Official test MAE of the attention model: I expect ____ (a single number or a range).
   Reason: ____ (hint: the noise floor is about 4 for sigma = 5, plus errors at event hours)
2. Versus last value (7.4): attention will be [better / about equal / worse] by about ____ %.
3. Versus same-hour-yesterday (12.9): attention will be [better / about equal / worse].
4. Attention weights: for Tuesday-Friday targets I expect the last row to put the most weight on
   position ____ (0 = same hour yesterday, 23 = previous hour). For Mon/Sat/Sun targets I expect ____.
5. Error at event hours: the model's MAE at event hours will be [lower / similar / higher] than at
   normal hours, because ____.
6. What would surprise me: ____.