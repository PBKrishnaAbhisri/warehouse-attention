"""Controls: does learned attention add value? Ridge, uniform attention, longer training."""
import numpy as np
from src.data_gen import generate, train_stats, make_windows
from src.baselines import baseline_predictions, mae
from experiments.warehouse import train_model, predict, metrics, pooled, print_table

series = generate(seed=0)
mean, std = train_stats(series)
train_w, val_w, test_w = [make_windows(series, s, mean, std) for s in ("train", "val", "test")]
control_w = pooled("test", range(1, 11), mean, std)


def features(w):   # flatten the 24x5 window and add the target hour's time features
    return np.concatenate([w["X"].reshape(len(w["y"]), -1), w["next_tf"]], axis=1)


def add_bias(F):
    return np.hstack([F, np.ones((len(F), 1))])


def fit_ridge(w, lam):
    F, y = add_bias(features(w)), (w["y"] - mean) / std
    P = lam * np.eye(F.shape[1])
    P[-1, -1] = 0                                        # do not penalise the bias
    return np.linalg.solve(F.T @ F + P, F.T @ y)


def ridge_predict(coef, w):
    return (add_bias(features(w)) @ coef) * std + mean


lams = [0.01, 0.1, 1, 10, 100]
best_lam = min(lams, key=lambda l: mae(ridge_predict(fit_ridge(train_w, l), val_w), val_w["y"]))
coef = fit_ridge(train_w, best_lam)
print("ridge lambda chosen on validation:", best_lam)

VARIANTS = {
    "attention, 60 epochs": dict(epochs=60),
    "attention, 200 epochs": dict(epochs=200),
    "uniform attention, 200 ep": dict(epochs=200, uniform=True),
}
trained = {name: [train_model(train_w, val_w, mean, std, seed=s, **kw)[0] for s in (0, 1, 2)]
           for name, kw in VARIANTS.items()}


def rows_for(w):
    rows = {m: metrics(p, w) for m, p in baseline_predictions(w["raw"]).items()}
    rows["ridge (linear)"] = metrics(ridge_predict(coef, w), w)
    for name, models in trained.items():
        per = [metrics(predict(m, w, mean, std)[0], w) for m in models]
        rows[name] = {k: float(np.mean([r[k] for r in per])) for k in per[0]}
    return rows


print_table("OFFICIAL TEST", rows_for(test_w), test_w)
print_table("POOLED CONTROL (seeds 1-10)", rows_for(control_w), control_w)
for name, models in trained.items():
    print(name, "official-test MAE per seed:",
          [round(metrics(predict(m, test_w, mean, std)[0], test_w)["mae"], 2) for m in models])