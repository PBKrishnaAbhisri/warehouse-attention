"""Synthetic warehouse demand generator (numpy only). Implements Phase 0, section 2.

Demand(t) = Base + Trend(t) + Daily(hour) + Weekly(day) + Event(t) + Noise(t), clipped at 0.
Day 0 of every series is a Monday.
"""
import numpy as np

DAILY_PEAKS = ((11.0, 40.0, 2.0), (16.0, 30.0, 2.0))      # (centre hour, height, width)
WEEKLY_OFFSET = np.array([0, 0, 0, 0, 0, -15, -25], dtype=float)   # Mon..Sun
SPIKE_TAU, SPIKE_LEN, DROP_LEN = 1.5, 6, 2
SPLITS = {"train": (0, 84), "val": (84, 102), "test": (102, 120)}  # day ranges


def generate(seed=0, n_days=120, base=100.0, trend=0.01, noise_sigma=5.0,
             p_spike=0.004, spike_scale=1.0, p_drop=0.002):
    rng = np.random.default_rng(seed)
    n = n_days * 24
    t = np.arange(n)
    hour, dow = t % 24, (t // 24) % 7

    daily = sum(h * np.exp(-((hour - c) ** 2) / (2 * w ** 2)) for c, h, w in DAILY_PEAKS)
    systematic = base + trend * t + daily + WEEKLY_OFFSET[dow]

    event = np.zeros(n)
    for s in np.flatnonzero(rng.random(n) < p_spike):           # spikes: jump, then decay
        size = rng.uniform(40, 80) * spike_scale
        lags = np.arange(min(SPIKE_LEN, n - s))
        event[s + lags] += size * np.exp(-lags / SPIKE_TAU)
    for s in np.flatnonzero(rng.random(n) < p_drop):            # drops: flat dip for 2 hours
        size = rng.uniform(30, 60)
        lags = np.arange(min(DROP_LEN, n - s))
        event[s + lags] -= size

    noise = rng.normal(0.0, noise_sigma, n)
    demand = np.clip(systematic + event + noise, 0.0, None)
    return dict(demand=demand, event=event, systematic=systematic, hour=hour, dow=dow)


def time_features(hour, dow):
    """sin/cos so that 23:00 is close to 00:00 and Sunday is close to Monday."""
    return np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                     np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], axis=-1)


def train_stats(series):
    d = series["demand"][: SPLITS["train"][1] * 24]
    return float(d.mean()), float(d.std())


def make_windows(series, split, mean, std, window=24):
    """Every window (24 inputs + 1 target) lies entirely inside one split."""
    lo, hi = (d * 24 for d in SPLITS[split])
    targets = np.arange(lo + window, hi)
    idx = targets[:, None] + np.arange(-window, 0)[None, :]            # (N, 24)
    raw = series["demand"][idx]
    tf = time_features(series["hour"], series["dow"])                  # (T, 4)
    X = np.concatenate([((raw - mean) / std)[..., None], tf[idx]], axis=-1)   # (N, 24, 5)
    return dict(X=X.astype(np.float32), next_tf=tf[targets].astype(np.float32),
                y=series["demand"][targets], raw=raw,
                event=series["event"][targets], target_idx=targets)