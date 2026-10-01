"""Simple baselines. raw[:, k] = demand at hour (t-23+k) for a window ending at t."""
import numpy as np


def baseline_predictions(raw):
    return {
        "last_value": raw[:, -1],               # y(t)
        "moving_average_24h": raw.mean(axis=1), # mean of the 24 inputs
        "same_hour_yesterday": raw[:, 0],       # y(t-23) = same hour of day as the target
    }


def mae(pred, y):
    return float(np.mean(np.abs(pred - y)))


def rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def event_mask(event, threshold=10.0):
    """Target hours where an event (spike/drop) moves demand by more than `threshold`."""
    return np.abs(event) > threshold