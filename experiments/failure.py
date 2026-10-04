"""Failure investigation: attention under larger spikes (the 'spikes x2' world).
Question: after a big spike, does the model pass the spike value forward into later predictions?"""
import numpy as np
from src.data_gen import generate, train_stats, make_windows
from src.baselines import mae
from src.ridge import fit_ridge, ridge_predict

KW = dict(spike_scale=2.0, p_spike=0.008)
GROUPS = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-6", 4, 6),
          ("7-12", 7, 12), ("13-24", 13, 24), ("none in window", 25, 10**9)]


def hours_since_big_event(event, thr=15.0):
    """lag[t] = hours since the last hour (before t) where the positive event effect exceeded thr."""
    lag = np.empty(len(event), dtype=np.int64)
    last = -10**6
    for t in range(len(event)):
        lag[t] = t - last
        if event[t] > thr:
            last = t
    return lag


def build(seeds, mean, std, **kw):
    out = []
    for s in seeds:
        series = generate(seed=s, **kw)
        w = make_windows(series, "test", mean, std)
        idx = w["target_idx"][:, None] + np.arange(-24, 0)[None, :]
        w["lag"] = hours_since_big_event(series["event"])[w["target_idx"]]
        w["event_win"] = series["event"][idx]            # event effect inside each input hour
        out.append(w)
    return {k: np.concatenate([w[k] for w in out]) for k in out[0]}


def without_events(w, mean, std):
    """Counterfactual input: the same window with the event effect removed from the demand inputs."""
    clean = dict(w)
    X = w["X"].copy()
    X[:, :, 0] = ((w["raw"] - w["event_win"] - mean) / std).astype(np.float32)
    clean["X"] = X
    return clean


if __name__ == "__main__":
    from experiments.warehouse import train_model, predict

    series = generate(seed=0)
    mean, std = train_stats(series)
    train_w = make_windows(series, "train", mean, std)
    val_w = make_windows(series, "val", mean, std)
    models = [train_model(train_w, val_w, mean, std, epochs=200, seed=s)[0] for s in (0, 1, 2)]
    lams = [0.01, 0.1, 1, 10, 100]
    lam = min(lams, key=lambda l: mae(ridge_predict(fit_ridge(train_w, l, mean, std),
                                                    val_w, mean, std), val_w["y"]))
    coef = fit_ridge(train_w, lam, mean, std)

    w = build(range(1, 11), mean, std, **KW)
    clean = without_events(w, mean, std)
    y = w["y"]
    preds = {"attention": [predict(m, w, mean, std)[0] for m in models],
             "ridge": [ridge_predict(coef, w, mean, std)]}
    preds_clean = {"attention": [predict(m, clean, mean, std)[0] for m in models],
                   "ridge": [ridge_predict(coef, clean, mean, std)]}
    A0 = predict(models[0], w, mean, std)[1][:, -1, :]           # last-row weights, seed 0: (N, 24)
    base_w = A0[w["lag"] > 24].mean(axis=0)                      # usual weight per position

    print("hours since the last big event (positive effect > 15) -> metrics, spikes x2 world")
    print("bias = mean(prediction - truth); effect = mean |prediction - prediction with events removed|")
    print(f"{'lag':15s} {'n':>5s} | {'att MAE':>7s} {'bias':>6s} {'effect':>6s} | "
          f"{'rdg MAE':>7s} {'bias':>6s} {'effect':>6s} | {'last MAE':>8s} | {'att weight on spike pos (x usual)':>34s}")
    for label, lo, hi in GROUPS:
        m = (w["lag"] >= lo) & (w["lag"] <= hi)
        cells = []
        for name in ("attention", "ridge"):
            e = np.mean([np.abs(p[m] - y[m]).mean() for p in preds[name]])
            b = np.mean([(p[m] - y[m]).mean() for p in preds[name]])
            f = np.mean([np.abs(p[m] - pc[m]).mean()
                         for p, pc in zip(preds[name], preds_clean[name])])
            cells.append(f"{e:7.2f} {b:6.2f} {f:6.2f}")
        last = np.abs(w["raw"][m, -1] - y[m]).mean()
        if hi <= 24:
            pos = 24 - w["lag"][m]                               # window index of the big-event hour
            ratio = (A0[m][np.arange(m.sum()), pos] / base_w[pos]).mean()
            rtxt = f"{ratio:34.2f}"
        else:
            rtxt = f"{'-':>34s}"
        print(f"{label:15s} {int(m.sum()):5d} | {cells[0]} | {cells[1]} | {last:8.2f} | {rtxt}")