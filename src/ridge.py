"""Ridge regression on the flattened 24x5 window + target-hour time features (a linear control)."""
import numpy as np


def _design(w):
    F = np.concatenate([w["X"].reshape(len(w["y"]), -1), w["next_tf"]], axis=1).astype(np.float64)
    return np.hstack([F, np.ones((len(F), 1))])          # last column = bias


def fit_ridge(w, lam, mean, std):
    F, y = _design(w), (w["y"] - mean) / std
    P = lam * np.eye(F.shape[1])
    P[-1, -1] = 0                                         # do not penalise the bias
    return np.linalg.solve(F.T @ F + P, F.T @ y)


def ridge_predict(coef, w, mean, std):
    return (_design(w) @ coef) * std + mean