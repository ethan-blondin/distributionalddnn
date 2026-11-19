"""Post-build checks: integrity, overlap consistency, daily alignment, summary stats."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (DAILY_COLUMNS, DATA_DIR, HOURS_PER_DAY, OVERLAP, REFERENCE_FILE,
                     TARGET_COLUMNS, WINDOWS)
from .parsing import read_reference
from .transforms import reference_daily

RATIO_TOLERANCE = (0.8, 1.25)   # mean ratio outside -> probable unit / currency issue


def load_outputs(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    out = {"DE.csv": read_reference(data_dir / REFERENCE_FILE)}
    for cfg in WINDOWS.values():
        out[cfg.output] = read_reference(data_dir / cfg.output)
    return out


# 1. Integrity -------------------------------------------------------------- #
def check_integrity(df: pd.DataFrame, expected_days: int) -> dict:
    idx = df.index
    steps = pd.Series(idx).diff().dropna()
    res = {
        "rows": len(df),
        "expected_rows": expected_days * HOURS_PER_DAY,
        "first": str(idx.min()),
        "last": str(idx.max()),
        "columns_ok": list(df.columns) == TARGET_COLUMNS,
        "monotonic_hourly": bool(idx.is_monotonic_increasing and (steps == pd.Timedelta(hours=1)).all()),
        "starts_00h": idx[0].hour == 0,
        "ends_23h": idx[-1].hour == 23,
        "duplicates": int(idx.duplicated().sum()),
        "nan": int(df.isna().sum().sum()),
        "inf": int(np.isinf(df.to_numpy()).sum()),
    }
    res["passed"] = (res["rows"] == res["expected_rows"] and res["columns_ok"] and res["monotonic_hourly"]
                     and res["starts_00h"] and res["ends_23h"]
                     and res["duplicates"] == res["nan"] == res["inf"] == 0)
    return res


# 2. Overlap consistency ---------------------------------------------------- #
def overlap_table(frames: dict[str, pd.DataFrame], period=OVERLAP) -> pd.DataFrame:
    """MAE and mean ratio per column for every pair of datasets on the overlap."""
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(hours=23)
    names = list(frames)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A = frames[a].loc[start:end]
            B = frames[b].loc[start:end]
            common = A.index.intersection(B.index)
            for col in TARGET_COLUMNS:
                x, y = A.loc[common, col], B.loc[common, col]
                ratio = x.mean() / y.mean() if y.mean() else np.nan
                rows.append({
                    "pair": f"{a} vs {b}", "column": col, "hours": len(common),
                    "MAE": (x - y).abs().mean(),
                    "rel_MAE_%": 100 * (x - y).abs().mean() / y.abs().mean(),
                    "mean_ratio": ratio,
                    "flag": "" if RATIO_TOLERANCE[0] <= ratio <= RATIO_TOLERANCE[1] else "CHECK UNITS",
                })
    return pd.DataFrame(rows)


# 3. Daily alignment (lag) check --------------------------------------------- #
def alignment_table(reference: pd.DataFrame, daily_eur: dict, has_quote: dict,
                    lags=range(-3, 4)) -> pd.DataFrame:
    """Which day shift of the raw daily series best matches DE.csv?

    Only days where the raw file has a real quote are compared. Best lag 0
    means DE.csv stores day T's close on day T (no lag baked in).
    """
    rows = []
    for col in DAILY_COLUMNS:
        ref = reference_daily(reference, col)
        raw, quote = daily_eur[col], has_quote[col]
        scores = {}
        for lag in lags:
            shifted = raw.shift(lag)
            mask = quote & shifted.notna()
            common = mask[mask].index.intersection(ref.index)
            if len(common) < 20:
                continue
            scores[lag] = ((shifted[common] - ref[common]).abs().mean() / ref[common].abs().mean(), len(common))
        if not scores:
            rows.append({"column": col, "days_compared": 0})
            continue
        best = min(scores, key=lambda k: scores[k][0])
        rows.append({"column": col, "days_compared": scores[0][1] if 0 in scores else None,
                     "rel_MAE_%_lag0": 100 * scores[0][0] if 0 in scores else None,
                     "best_lag_days": best, "rel_MAE_%_best": 100 * scores[best][0]})
    return pd.DataFrame(rows)


# 4. Summary statistics ----------------------------------------------------- #
def summary_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name, df in frames.items():
        st = df[TARGET_COLUMNS].agg(["min", "median", "mean", "max", "std"]).T
        st.insert(0, "dataset", name)
        parts.append(st)
    return pd.concat(parts).rename_axis("column").reset_index().set_index(["column", "dataset"]).sort_index(
        level=0, sort_remaining=False).reindex(TARGET_COLUMNS, level=0)


def validate_outputs(data_dir: Path = DATA_DIR, verbose: bool = True) -> dict:
    """Run all checks on the written CSVs. Raises AssertionError on integrity failure."""
    frames = load_outputs(data_dir)
    integrity = {cfg.output: check_integrity(frames[cfg.output], cfg.expected_days) for cfg in WINDOWS.values()}
    overlap = overlap_table(frames)
    summary = summary_table(frames)

    if verbose:
        with pd.option_context("display.width", 200, "display.max_columns", 20,
                               "display.float_format", "{:,.3f}".format):
            print("=" * 80, "\n1. INTEGRITY\n", "=" * 80, sep="")
            print(pd.DataFrame(integrity).T.to_string())
            print("\n", "=" * 80, f"\n2. OVERLAP {OVERLAP[0]} .. {OVERLAP[1]}\n", "=" * 80, sep="")
            print("Note: 2015-20 power columns come from DE.csv, 2020-26 from SMARD -> "
                  "small differences are expected; 'CHECK UNITS' flags ratio outside "
                  f"{RATIO_TOLERANCE}.")
            print(overlap.to_string(index=False))
            print("\n", "=" * 80, "\n3. SUMMARY STATISTICS\n", "=" * 80, sep="")
            print(summary.to_string())

    failed = [k for k, v in integrity.items() if not v["passed"]]
    assert not failed, f"Integrity checks failed for {failed}"
    return {"integrity": integrity, "overlap": overlap, "summary": summary}
