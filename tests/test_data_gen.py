import numpy as np
from src.data_gen import generate, make_windows, train_stats, SPLITS


def test_same_seed_gives_identical_data():
    a, b = generate(seed=3), generate(seed=3)
    assert np.array_equal(a["demand"], b["demand"])
    assert not np.array_equal(a["demand"], generate(seed=4)["demand"])


def test_length_and_non_negative():
    s = generate(seed=0)
    assert len(s["demand"]) == 120 * 24
    assert (s["demand"] >= 0).all()


def test_noise_matches_configured_sigma():
    s = generate(seed=0, noise_sigma=5.0)
    resid = s["demand"] - s["systematic"] - s["event"]
    resid = resid[s["demand"] > 0]                       # ignore clipped hours
    assert abs(resid.std() - 5.0) < 0.5 and abs(resid.mean()) < 0.5


def test_daily_pattern_has_morning_peak():
    s = generate(seed=0)
    h = s["hour"]
    assert s["demand"][h == 11].mean() > s["demand"][h == 3].mean() + 30


def test_windows_stay_inside_their_split_and_have_right_shape():
    s = generate(seed=0)
    mean, std = train_stats(s)
    for split, (d0, d1) in SPLITS.items():
        w = make_windows(s, split, mean, std)
        assert w["X"].shape == ((d1 - d0) * 24 - 24, 24, 5)
        assert w["target_idx"].min() - 24 >= d0 * 24       # window start inside the split
        assert w["target_idx"].max() < d1 * 24             # target inside the split


def test_window_contents_are_the_24_previous_hours():
    s = generate(seed=0)
    mean, std = train_stats(s)
    w = make_windows(s, "test", mean, std)
    t = w["target_idx"]
    assert np.array_equal(w["y"], s["demand"][t])
    assert np.array_equal(w["raw"][:, -1], s["demand"][t - 1])    # last input = hour t-1
    assert np.array_equal(w["raw"][:, 0], s["demand"][t - 24])    # first input = same hour yesterday