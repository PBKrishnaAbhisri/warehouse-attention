"""Evaluate the baselines on the official test split and on the shifted conditions."""
import numpy as np
from src.data_gen import generate, train_stats, make_windows
from src.baselines import baseline_predictions, mae, rmse, event_mask

orig = generate(seed=0)
mean, std = train_stats(orig)          # normalisation from the ORIGINAL training days only

CONDITIONS = {                         # pooled over seeds 1..10 for stable event statistics
    "control (same params)": dict(),
    "noise x2": dict(noise_sigma=10.0),
    "spikes x2 (size, freq)": dict(spike_scale=2.0, p_spike=0.008),
}


def pooled(split, seeds, **kw):
    ws = [make_windows(generate(seed=s, **kw), split, mean, std) for s in seeds]
    return {k: np.concatenate([w[k] for w in ws]) for k in ws[0]}


def report(name, w):
    ev = event_mask(w["event"])
    print(f"\n{name}  (n={len(w['y'])}, event hours={int(ev.sum())})")
    print(f"{'method':22s} {'MAE':>7s} {'RMSE':>7s} {'MAE@events':>11s}")
    for m, p in baseline_predictions(w["raw"]).items():
        e = mae(p[ev], w["y"][ev]) if ev.any() else float("nan")
        print(f"{m:22s} {mae(p, w['y']):7.2f} {rmse(p, w['y']):7.2f} {e:11.2f}")


if __name__ == "__main__":
    report("official val", make_windows(orig, "val", mean, std))
    report("official test", make_windows(orig, "test", mean, std))
    for name, kw in CONDITIONS.items():
        report(f"test, {name}", pooled("test", range(1, 11), **kw))