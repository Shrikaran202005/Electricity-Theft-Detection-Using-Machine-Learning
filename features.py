"""Feature engineering: turn a daily-consumption series into theft-sensitive features."""
import numpy as np
import pandas as pd

DAY_COLS_PREFIX = "d"


def day_columns(df):
    return [c for c in df.columns if c.startswith(DAY_COLS_PREFIX) and c[1:].isdigit()]


def _row_features(x):
    n = len(x)
    mean = x.mean()
    std = x.std()
    med = np.median(x)
    h1, h2 = x[: n // 2], x[n // 2:]
    q = np.array_split(x, 4)
    qm = [c.mean() for c in q]

    # monthly means (12 chunks) -> trend slope + volatility
    months = np.array([c.mean() for c in np.array_split(x, 12)])
    slope = np.polyfit(np.arange(12), months, 1)[0]

    # seasonality strength: how well a 1-yr sine explains the series (honest users show summer peak)
    t = np.arange(n)
    design = np.c_[np.ones(n), np.sin(2 * np.pi * t / n), np.cos(2 * np.pi * t / n)]
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    fit = design @ coef
    ss_res = ((x - fit) ** 2).sum()
    ss_tot = ((x - mean) ** 2).sum() + 1e-9
    seasonal_r2 = 1 - ss_res / ss_tot

    low_thr = 0.3 * mean
    below = x < low_thr
    # longest run of consecutive "very low" days
    run = best = 0
    for b in below:
        run = run + 1 if b else 0
        best = max(best, run)

    diffs = np.abs(np.diff(x))
    return {
        "mean_kwh": mean,
        "std_kwh": std,
        "cv": std / (mean + 1e-9),
        "median_kwh": med,
        "min_kwh": x.min(),
        "max_kwh": x.max(),
        "skew": pd.Series(x).skew(),
        "kurtosis": pd.Series(x).kurt(),
        "h2_over_h1": h2.mean() / (h1.mean() + 1e-9),
        "q4_over_q1": qm[3] / (qm[0] + 1e-9),
        "trend_slope": slope / (mean + 1e-9),
        "monthly_cv": months.std() / (months.mean() + 1e-9),
        "seasonal_r2": seasonal_r2,
        "low_day_frac": below.mean(),
        "zero_day_frac": (x < 0.05 * (mean + 1e-9)).mean(),
        "longest_low_run": best,
        "mean_abs_diff": diffs.mean() / (mean + 1e-9),
        "mean_median_gap": (mean - med) / (mean + 1e-9),
    }


def build_features(df):
    cols = day_columns(df)
    X = df[cols].to_numpy(dtype=float)
    feats = pd.DataFrame([_row_features(r) for r in X], index=df.index)
    # peer-relative features (compare customer against the whole population)
    feats["mean_vs_peer"] = feats["mean_kwh"] / (feats["mean_kwh"].median() + 1e-9)
    feats["cv_vs_peer"] = feats["cv"] / (feats["cv"].median() + 1e-9)
    return feats


FEATURE_NAMES = None


def feature_names():
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        FEATURE_NAMES = list(build_features(_dummy()).columns)
    return FEATURE_NAMES


def _dummy():
    rng = np.random.default_rng(0)
    d = pd.DataFrame(rng.uniform(1, 5, (3, 365)), columns=[f"d{i+1:03d}" for i in range(365)])
    return d
