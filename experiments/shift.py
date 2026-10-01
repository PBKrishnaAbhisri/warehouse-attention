"""Distribution-shift experiment (Phase 0 section 8, hypothesis H3).
Models are trained ONLY on the original distribution, then tested on shifted data."""
import json
import numpy as np
from src.data_gen import generate, train_stats, make_windows
from src.baselines import baseline_predictions, mae, event_mask
from src.ridge import fit_ridge, ridge_predict
from experiments.warehouse import train_model, predict, pooled

series = generate(seed=0)
mean, std = train_stats(series)                 # original training statistics, never re-fitted
train_w = make_windows(series, "train", mean, std)
val_w = make_windows(series, "val", mean, std)

CONDITIONS = {
    "control (original)": dict(),
    "noise x2 (sigma 10)": dict(noise_sigma=10.0),
    "spikes x2 (size, freq)": dict(spike_scale=2.0, p_spike=0.008),
}

models = [train_model(train_w, val_w, mean, std, epochs=200, seed=s)[0] for s in (0, 1, 2)]
lams = [0.01, 0.1, 1, 10, 100]
best_lam = min(lams, key=lambda l: mae(ridge_predict(fit_ridge(train_w, l, mean, std),
                                                     val_w, mean, std), val_w["y"]))
coef = fit_ridge(train_w, best_lam, mean, std)
print("ridge lambda (chosen on validation):", best_lam)


def stats(p, w):
    ev = event_mask(w["event"])
    err = np.abs(p - w["y"])
    return dict(mae=float(err.mean()), rmse=float(np.sqrt(((p - w["y"]) ** 2).mean())),
                mae_normal=float(err[~ev].mean()), mae_events=float(err[ev].mean()))


def all_stats(w):
    rows = {m: stats(p, w) for m, p in baseline_predictions(w["raw"]).items()}
    rows["ridge"] = stats(ridge_predict(coef, w, mean, std), w)
    per = [stats(predict(m, w, mean, std)[0], w) for m in models]
    rows["attention"] = {k: float(np.mean([r[k] for r in per])) for k in per[0]}
    return rows


W = {name: pooled("test", range(1, 11), mean, std, **kw) for name, kw in CONDITIONS.items()}
results = {}
for name, w in W.items():
    results[name] = all_stats(w)
    ev = event_mask(w["event"])
    print(f"\n{name}  (windows {len(w['y'])}, event hours {int(ev.sum())})")
    print(f"{'method':22s} {'MAE':>7} {'RMSE':>7} {'MAE normal':>11} {'MAE events':>11}")
    for m, r in results[name].items():
        print(f"{m:22s} {r['mae']:7.2f} {r['rmse']:7.2f} {r['mae_normal']:11.2f} {r['mae_events']:11.2f}")

base = results["control (original)"]
print("\nChange vs control (%):   overall | normal hours | event hours")
for name in list(CONDITIONS)[1:]:
    print(name)
    for m, r in results[name].items():
        d = lambda k: 100 * (r[k] / base[m][k] - 1)
        print(f"  {m:22s} {d('mae'):6.0f}% | {d('mae_normal'):6.0f}% | {d('mae_events'):6.0f}%")

print("\nAttention of the last row (model seed 0): windows with a very large input vs the rest")
for name, w in W.items():
    _, A = predict(models[0], w, mean, std)
    a = A[:, -1, :]
    ent = -(a * np.log(a + 1e-12)).sum(axis=1)
    big = w["X"][:, :, 0].max(axis=1) > 3.0           # some input > 3 std above the training mean
    e_big = ent[big].mean() if big.any() else float("nan")
    print(f"{name:24s} big-input windows: {int(big.sum()):4d}  entropy {e_big:.3f}"
          f"  | other windows entropy {ent[~big].mean():.3f}   (uniform = {np.log(24):.3f})")

with open("results/shift_results.json", "w") as f:
    json.dump(results, f, indent=2)