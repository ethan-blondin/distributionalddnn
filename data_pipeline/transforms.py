"""Calendar alignment, DST normalisation, gap filling and daily->hourly broadcast."""
from __future__ import annotations

import pandas as pd

from .config import HOURS_PER_DAY

WEEK = 7 * HOURS_PER_DAY


def hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    """24 naive hourly stamps per calendar day, start 00:00 .. end 23:00."""
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")


def daily_index(start: str, end: str, lookback_days: int = 0) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=lookback_days),
                         pd.Timestamp(end), freq="D")


# --------------------------------------------------------------------------- #
# Hourly (SMARD) series
# --------------------------------------------------------------------------- #
def normalize_hourly(s: pd.Series, index: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    """Force a local-time hourly series onto a strict 24 h/day grid.

    * Fall-back (25 h day): the two 02:00 observations are averaged.
    * Spring-forward (23 h day): the missing 02:00 is created as NaN and later
      linearly interpolated by fill_hourly_gaps() (= mean of 01:00 and 03:00).
    """
    if s.index.has_duplicates:
        duplicated_hours = int(s.index.duplicated().sum())
    else:
        duplicated_hours = 0
    # Only hourly data is supported (averaging quarter-hours of MWh would be wrong).
    step = pd.Series(s.index.unique().sort_values()).diff().mode().iloc[0]
    if step != pd.Timedelta(hours=1):
        raise ValueError(f"Expected hourly resolution, got {step}")

    merged = s.groupby(level=0).mean()
    out = merged.reindex(index)
    report = {
        "dst_duplicates_averaged": duplicated_hours,
        "hours_missing_on_grid": int(out.index.difference(merged.index).size),
    }
    return out, report


def fill_hourly_gaps(s: pd.Series, max_interp_hours: int = 3) -> tuple[pd.Series, dict]:
    """Fill NaN in an hourly series.

    * gaps <= max_interp_hours (DST hole, single '-' cells): linear interpolation
    * longer gaps (whole missing days in SMARD): value of the same hour one week
      earlier, else one week later (weekly profile is preserved, unlike a straight
      line over 24 h); anything left is interpolated.
    """
    na = s.isna()
    report = {"nan_before": int(na.sum()), "interpolated": 0, "weekly_profile": 0, "residual_interp": 0}
    if not na.any():
        return s, report

    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform("sum")
    short = na & (run_len <= max_interp_hours)

    out = s.copy()
    interp = s.interpolate(method="linear", limit_area="inside")
    out[short] = interp[short]
    report["interpolated"] = int(out[short].notna().sum())

    before = out.isna().sum()
    for k in (WEEK, -WEEK, 2 * WEEK, -2 * WEEK):
        out = out.fillna(out.shift(k))
    report["weekly_profile"] = int(before - out.isna().sum())

    left = int(out.isna().sum())
    if left:
        out = out.interpolate(method="linear", limit_direction="both")
        report["residual_interp"] = left
    return out, report


# --------------------------------------------------------------------------- #
# Daily series
# --------------------------------------------------------------------------- #
def to_calendar(s: pd.Series, days: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Put a sparse daily series on a continuous calendar and forward-fill.

    Observations before the calendar start are used to seed the forward fill.
    Returns (filled, has_quote) where has_quote marks days with a real quote.
    """
    has_quote = s.reindex(days).notna()
    full = s.reindex(s.index.union(days)).ffill().reindex(days)
    return full, has_quote


def broadcast_daily(daily: pd.Series, hours: pd.DatetimeIndex) -> pd.Series:
    """Repeat each day's value on its 24 hours (00:00..23:00), no lag applied."""
    return pd.Series(daily.reindex(hours.normalize()).to_numpy(), index=hours, name=daily.name)


def reference_daily(reference: pd.DataFrame, column: str) -> pd.Series:
    """Daily value of a broadcast column of DE.csv (value at 00:00 of each day)."""
    s = reference[column]
    return s.groupby(s.index.normalize()).first()
