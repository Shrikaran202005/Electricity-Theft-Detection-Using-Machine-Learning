"""Synthetic smart-meter dataset generator for electricity theft detection.

Each customer gets 24 months of monthly-ish daily readings summarised into a
daily kWh series. Honest customers follow a seasonal, weekly-patterned load.
Thieves are injected using the classic attack patterns from the literature
(e.g. the State Grid Corporation of China theft benchmark):

    1. scale_down   - consumption multiplied by a constant factor < 1
    2. random_scale - each day multiplied by a different random factor
    3. night_zero   - readings dropped to ~0 for a block of hours (daily-level: some days zeroed)
    4. gradual_drop - consumption fades progressively (meter tampering)
    5. intermittent - random days reported as zero (bypass connection)
"""
import numpy as np
import pandas as pd

N_DAYS = 365
RNG_SEED = 42


def _honest_profile(rng, n_days):
    base = rng.uniform(4, 25)  # kWh/day household baseline
    days = np.arange(n_days)
    seasonal = 1 + 0.30 * np.sin(2 * np.pi * (days - 90) / 365)   # summer peak (AC load)
    weekly = 1 + 0.08 * np.sin(2 * np.pi * days / 7)
    noise = rng.normal(1.0, 0.10, n_days)
    series = base * seasonal * weekly * noise
    return np.clip(series, 0.1, None)


def _apply_theft(series, kind, rng):
    s = series.copy()
    n = len(s)
    if kind == "scale_down":
        s *= rng.uniform(0.2, 0.7)
    elif kind == "random_scale":
        s *= rng.uniform(0.2, 0.9, n)
    elif kind == "gradual_drop":
        start = rng.integers(int(0.08 * n), int(0.55 * n))
        fade = np.ones(n)
        fade[start:] = np.linspace(1.0, rng.uniform(0.15, 0.5), n - start)
        s *= fade
    elif kind == "intermittent":
        mask = rng.random(n) < rng.uniform(0.15, 0.40)
        s[mask] = rng.uniform(0.0, 0.3, mask.sum())
    elif kind == "night_zero":
        start = rng.integers(int(0.05 * n), int(0.68 * n))
        length = rng.integers(int(0.11 * n), int(0.30 * n))
        s[start:start + length] *= rng.uniform(0.0, 0.25)
    return np.clip(s, 0, None)


THEFT_KINDS = ["scale_down", "random_scale", "gradual_drop", "intermittent", "night_zero"]


def generate(n_customers=1200, theft_ratio=0.12, n_days=N_DAYS, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n_theft = int(n_customers * theft_ratio)
    labels = np.zeros(n_customers, dtype=int)
    labels[rng.choice(n_customers, n_theft, replace=False)] = 1

    rows, kinds = [], []
    for i in range(n_customers):
        s = _honest_profile(rng, n_days)
        kind = "none"
        if labels[i]:
            kind = rng.choice(THEFT_KINDS)
            s = _apply_theft(s, kind, rng)
        rows.append(s)
        kinds.append(kind)

    cols = [f"d{d+1:03d}" for d in range(n_days)]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "customer_id", [f"C{i+1:05d}" for i in range(n_customers)])
    df["theft_type"] = kinds   # ground truth (for evaluation only)
    df["is_theft"] = labels
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/smart_meter_data.csv", index=False)
    print(df.shape, "theft rate:", df.is_theft.mean().round(3))
    print(df.theft_type.value_counts())
