"""Daily design matrix (227 inputs) and targets, built once per dataset.

Row d of X holds the information set available on day d-1 (11:30 CET) for
forecasting the 24 prices of day d:

    cols   0- 23  price D-1        cols 120-143  load  D-1
    cols  24- 47  price D-2        cols 144-167  load  D-7
    cols  48- 71  price D-3        cols 168-191  RES   D
    cols  72- 95  price D-7        cols 192-215  RES   D-1
    cols  96-119  load  D          cols 216-219  EUA, Coal, Gas, Oil at D-2
                                   cols 220-226  day-of-week one-hot of day D

Rows d < 7 are undefined (NaN) because D-7 lags do not exist.
All three bugs of the original code are fixed by construction: features are
indexed by absolute day number (no window-relative `data` vs `df` mix-up) and
the weekday is taken from the day's first timestamp and one-hot encoded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MAX_LAG, N_DAYS

N_INPUTS = 227

# Optuna parameter name -> column slice (the 14 input groups of the paper).
FEATURE_GROUPS = {
    "price_D-1": slice(0, 24), "price_D-2": slice(24, 48), "price_D-3": slice(48, 72),
    "price_D-7": slice(72, 96), "load_D": slice(96, 120), "load_D-1": slice(120, 144),
    "load_D-7": slice(144, 168), "RES_D": slice(168, 192), "RES_D-1": slice(192, 216),
    "EUA": slice(216, 217), "Coal": slice(217, 218), "Gas": slice(218, 219), "Oil": slice(219, 220),
    "Dummy": slice(220, 227),
}


@dataclass
class DailyData:
    dates: pd.DatetimeIndex   # one entry per day (00:00)
    X: np.ndarray             # (n_days, 227)
    Y: np.ndarray             # (n_days, 24) realised prices
    frame: pd.DataFrame       # hourly source data


def load_hourly(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
    if len(df) != N_DAYS * 24:
        raise ValueError(f"{path}: expected {N_DAYS * 24} rows, got {len(df)}")
    return df


def build_daily(df: pd.DataFrame) -> DailyData:
    n = len(df) // 24
    daily = lambda col: df[col].to_numpy(dtype="float64").reshape(n, 24)   # noqa: E731
    price, load, res = daily("Price"), daily("Load_DA_Forecast"), daily("Renewables_DA_Forecast")
    first_hour = lambda col: df[col].to_numpy(dtype="float64")[::24]        # noqa: E731
    fuels = np.column_stack([first_hour(c) for c in ("EUA", "API2_Coal", "TTF_Gas", "Brent_oil")])
    dates = df.index[::24]

    X = np.full((n, N_INPUTS), np.nan)
    d = np.arange(MAX_LAG, n)
    X[d, 0:24] = price[d - 1]
    X[d, 24:48] = price[d - 2]
    X[d, 48:72] = price[d - 3]
    X[d, 72:96] = price[d - 7]
    X[d, 96:120] = load[d]
    X[d, 120:144] = load[d - 1]
    X[d, 144:168] = load[d - 7]
    X[d, 168:192] = res[d]
    X[d, 192:216] = res[d - 1]
    X[d, 216:220] = fuels[d - 2]
    X[d, 220:227] = np.eye(7)[dates.dayofweek.to_numpy()[d]]
    return DailyData(dates=pd.DatetimeIndex(dates), X=X, Y=price, frame=df)


def column_mask(params: dict) -> np.ndarray:
    """Boolean mask over the 227 inputs from Optuna feature-selection flags."""
    mask = np.zeros(N_INPUTS, dtype=bool)
    for name, sl in FEATURE_GROUPS.items():
        if params.get(name, False):
            mask[sl] = True
    return mask
