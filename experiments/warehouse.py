"""Train the attention warehouse model and compare it with the baselines on the same windows."""
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data_gen import generate, train_stats, make_windows
from src.baselines import baseline_predictions, mae, rmse, event_mask
from src.models import WarehouseModel


def to_tensors(w, mean, std):
    return (torch.from_numpy(w["X"]), torch.from_numpy(w["next_tf"]),
            torch.from_numpy(((w["y"] - mean) / std).astype(np.float32)))


def predict(model, w, mean, std):
    X, tf, _ = to_tensors(w, mean, std)
    model.eval()
    with torch.no_grad():
        out, A = model(X, tf)
    return out.numpy() * std + mean, A.numpy()          # back to orders per hour


def train_model(train_w, val_w, mean, std, d_k=16, scale=True,
                lr=3e-3, epochs=60, batch=64, seed=0, uniform=False):
    torch.manual_seed(seed)
    model = WarehouseModel(d_k=d_k, scale=scale, uniform=uniform)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X, tf, y = to_tensors(train_w, mean, std)
    g = torch.Generator().manual_seed(seed)
    best, best_state, hist = float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(y), generator=g)
        total = 0.0
        for i in range(0, len(y), batch):
            b = perm[i:i + batch]
            out, _ = model(X[b], tf[b])
            loss = F.mse_loss(out, y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        val_mae = mae(predict(model, val_w, mean, std)[0], val_w["y"])
        hist.append(dict(epoch=epoch, train_mse=total / len(y), val_mae=val_mae))
        if val_mae < best:                              # keep the best epoch by VALIDATION error
            best = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


def metrics(p, w):
    ev = event_mask(w["event"])
    return dict(mae=mae(p, w["y"]), rmse=rmse(p, w["y"]),
                mae_events=mae(p[ev], w["y"][ev]) if ev.any() else float("nan"))


def evaluate_all(w, models, mean, std):
    rows = {m: metrics(p, w) for m, p in baseline_predictions(w["raw"]).items()}
    per_seed = [metrics(predict(m, w, mean, std)[0], w) for m in models]
    rows["attention (mean of seeds)"] = {k: float(np.mean([r[k] for r in per_seed]))
                                         for k in per_seed[0]}
    return rows, per_seed


def print_table(title, rows, w):
    print(f"\n{title}  (windows: {len(w['y'])}, event hours: {int(event_mask(w['event']).sum())})")
    print(f"{'method':28s} {'MAE':>7s} {'RMSE':>7s} {'MAE@events':>11s}")
    for name, r in rows.items():
        print(f"{name:28s} {r['mae']:7.2f} {r['rmse']:7.2f} {r['mae_events']:11.2f}")


def pooled(split, seeds, mean, std, **kw):
    ws = [make_windows(generate(seed=s, **kw), split, mean, std) for s in seeds]
    return {k: np.concatenate([w[k] for w in ws]) for k in ws[0]}


if __name__ == "__main__":
    series = generate(seed=0)
    mean, std = train_stats(series)
    train_w = make_windows(series, "train", mean, std)
    val_w = make_windows(series, "val", mean, std)
    test_w = make_windows(series, "test", mean, std)

    models = []
    for seed in (0, 1, 2):
        model, hist = train_model(train_w, val_w, mean, std, seed=seed)
        models.append(model)
        best_epoch = min(hist, key=lambda h: h["val_mae"])
        print(f"seed {seed}: best epoch {best_epoch['epoch']}, val MAE {best_epoch['val_mae']:.2f}, "
              f"test MAE {mae(predict(model, test_w, mean, std)[0], test_w['y']):.2f}")

    rows_test, per_seed = evaluate_all(test_w, models, mean, std)
    print_table("OFFICIAL TEST", rows_test, test_w)
    print("attention test MAE per seed:", [round(r["mae"], 2) for r in per_seed])

    control_w = pooled("test", range(1, 11), mean, std)
    rows_ctrl, _ = evaluate_all(control_w, models, mean, std)
    print_table("POOLED CONTROL (seeds 1-10, same parameters)", rows_ctrl, control_w)

    # Where does the last hour's row put its attention? (a visualisation, not a causal claim)
    _, A = predict(models[0], test_w, mean, std)
    last = A[:, -1, :]                                   # (N, 24)
    dow = (test_w["target_idx"] // 24) % 7
    groups = {"Tue-Fri targets": np.isin(dow, [1, 2, 3, 4]),
              "Mon/Sat/Sun targets": np.isin(dow, [0, 5, 6])}
    plt.figure(figsize=(8, 4))
    for name, mask in groups.items():
        avg = last[mask].mean(axis=0)
        print(f"{name}: strongest position = {int(avg.argmax())} (weight {avg.max():.3f}); "
              f"same-hour-yesterday (pos 0) = {avg[0]:.3f}; last hour (pos 23) = {avg[23]:.3f}")
        plt.plot(range(24), avg, marker="o", label=name)
    plt.axhline(1 / 24, color="gray", linestyle="--", label="uniform (1/24)")
    plt.xlabel("position in window (0 = same hour yesterday, 23 = previous hour)")
    plt.ylabel("mean attention weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/attention_weights.png", dpi=120)

    with open("results/warehouse_results.json", "w") as f:
        json.dump({"official_test": rows_test, "pooled_control": rows_ctrl}, f, indent=2)