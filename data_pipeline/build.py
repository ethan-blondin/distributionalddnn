"""Assemble one hourly dataset (DE.csv schema) for a configured window."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import parsing, transforms
from .config import (DAILY_COLUMNS, DAILY_SOURCES, DATA_DIR, FX_FILE, INDEX_NAME, POWER_COLUMNS,
                     REFERENCE_FILE, SMARD_SOURCES, TARGET_COLUMNS, TIMESTAMP_FORMAT, WindowConfig)


@dataclass
class BuildResult:
    window: WindowConfig
    data: pd.DataFrame                      # final hourly dataset
    report: dict = field(default_factory=dict)          # per-column provenance / fill counts
    daily_eur: dict = field(default_factory=dict)       # raw daily series in EUR (calendar, ffilled)
    daily_has_quote: dict = field(default_factory=dict)  # True where the raw file had a quote


def locate_fx_file(data_dir: Path) -> Path:
    """EUR/USD file: Datasets/ first, then any sub-folder."""
    direct = data_dir / FX_FILE
    if direct.exists():
        return direct
    hits = sorted(data_dir.rglob(FX_FILE))
    if not hits:
        raise FileNotFoundError(f"{FX_FILE} not found under {data_dir}")
    return hits[0]


def load_reference(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return parsing.read_reference(data_dir / REFERENCE_FILE)


def load_fx(data_dir: Path = DATA_DIR) -> pd.Series:
    """Daily EUR/USD close = USD per 1 EUR."""
    return parsing.read_investing(locate_fx_file(data_dir), "Price")


# --------------------------------------------------------------------------- #
# Power columns
# --------------------------------------------------------------------------- #
def power_from_reference(reference: pd.DataFrame, hours: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    sub = reference.reindex(hours)[POWER_COLUMNS]
    missing = sub.isna().sum()
    if missing.any():
        raise ValueError(f"DE.csv does not fully cover the window: missing {missing.to_dict()}")
    return sub, {c: {"source": REFERENCE_FILE} for c in POWER_COLUMNS}


def power_from_smard(folder: Path, hours: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    out, report = {}, {}
    for col, src in SMARD_SOURCES.items():
        path = parsing.find_file(folder, src.pattern)
        raw = parsing.read_smard(path, src.column_key, src.exclude_key)
        grid, dst_rep = transforms.normalize_hourly(raw, hours)
        filled, gap_rep = transforms.fill_hourly_gaps(grid)
        out[col] = filled
        report[col] = {"source": f"{path.name} :: {raw.name}", **dst_rep, **gap_rep}
    return pd.DataFrame(out, index=hours), report


# --------------------------------------------------------------------------- #
# Daily commodity columns
# --------------------------------------------------------------------------- #
def daily_commodities(cfg: WindowConfig, folder: Path, fx: pd.Series, reference: pd.DataFrame | None,
                      hours: pd.DatetimeIndex):
    days = transforms.daily_index(cfg.start, cfg.end, cfg.lookback_days)
    window_days = transforms.daily_index(cfg.start, cfg.end)
    fx_cal, _ = transforms.to_calendar(fx, days)

    hourly, report, daily_eur, has_quote_all = {}, {}, {}, {}
    for col, src in DAILY_SOURCES.items():
        path = parsing.find_file(folder, src.pattern)
        raw = (parsing.read_investing(path, src.value_column) if src.kind == "investing"
               else parsing.read_fred(path, src.value_column))

        # 1) calendar + ffill of the native-currency price  2) FX conversion afterwards
        cal, has_quote = transforms.to_calendar(raw, days)
        if src.currency == "USD":
            cal = cal / fx_cal
        elif src.currency != "EUR":
            raise ValueError(f"Unsupported currency {src.currency}")

        daily = cal.reindex(window_days)
        quote = has_quote.reindex(window_days)
        rep = {
            "source": path.name,
            "currency": f"{src.currency}->EUR" if src.currency == "USD" else "EUR",
            "raw_first_quote": raw.index.min().date().isoformat(),
            "raw_last_quote": raw.index.max().date().isoformat(),
            "days_with_quote": int(quote.sum()),
            "days_ffilled": int((~quote & daily.notna()).sum()),
            "days_from_reference": 0,
            "days_bfilled": 0,
        }

        # 3) days before the first raw quote: DE.csv if allowed, else back-fill
        if daily.isna().any() and cfg.reference_fallback and reference is not None:
            ref = transforms.reference_daily(reference, col).reindex(window_days)
            take = daily.isna() & ref.notna()
            daily[take] = ref[take]
            rep["days_from_reference"] = int(take.sum())
        if daily.isna().any():
            rep["days_bfilled"] = int(daily.isna().sum())
            daily = daily.bfill().ffill()

        daily.name = col
        hourly[col] = transforms.broadcast_daily(daily, hours)
        report[col] = rep
        daily_eur[col] = cal
        has_quote_all[col] = has_quote
    return pd.DataFrame(hourly, index=hours), report, daily_eur, has_quote_all


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_window(cfg: WindowConfig, data_dir: Path = DATA_DIR,
                 reference: pd.DataFrame | None = None, fx: pd.Series | None = None) -> BuildResult:
    data_dir = Path(data_dir)
    folder = data_dir / cfg.folder
    reference = load_reference(data_dir) if reference is None else reference
    fx = load_fx(data_dir) if fx is None else fx
    hours = transforms.hourly_index(cfg.start, cfg.end)

    if cfg.power_source == "reference":
        power, power_rep = power_from_reference(reference, hours)
    elif cfg.power_source == "smard":
        power, power_rep = power_from_smard(folder, hours)
    else:
        raise ValueError(f"Unknown power_source {cfg.power_source!r}")

    daily, daily_rep, daily_eur, has_quote = daily_commodities(cfg, folder, fx, reference, hours)

    df = pd.concat([power, daily], axis=1)[TARGET_COLUMNS].astype("float64")
    df.index.name = INDEX_NAME
    return BuildResult(cfg, df, {**power_rep, **daily_rep}, daily_eur, has_quote)


def write_dataset(result: BuildResult, data_dir: Path = DATA_DIR) -> Path:
    path = Path(data_dir) / result.window.output
    result.data.to_csv(path, date_format=TIMESTAMP_FORMAT)
    return path


def report_frame(result: BuildResult) -> pd.DataFrame:
    """Provenance / filling report as a table (one row per column)."""
    return pd.DataFrame(result.report).T.reindex(TARGET_COLUMNS).replace({np.nan: ""})
