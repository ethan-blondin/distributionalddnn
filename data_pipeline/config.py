"""Static configuration of the dataset-preparation pipeline.

Everything that describes *what* to build (windows, file patterns, column
names, units) lives here, so the processing modules only describe *how*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repository root = parent of this package; raw data lives in <root>/Datasets.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Datasets"

REFERENCE_FILE = "DE.csv"              # golden dataset of Marcjasz et al. (2022)
FX_FILE = "EUR_USD Historical Data.csv"  # daily EUR/USD (USD per 1 EUR)

HOURS_PER_DAY = 24
INDEX_NAME = "date_time"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Target schema: exact column order of Datasets/DE.csv.
POWER_COLUMNS = ["Price", "Load_DA_Forecast", "Renewables_DA_Forecast"]
DAILY_COLUMNS = ["EUA", "API2_Coal", "TTF_Gas", "Brent_oil"]
TARGET_COLUMNS = POWER_COLUMNS + DAILY_COLUMNS


@dataclass(frozen=True)
class SmardSource:
    """One SMARD hourly export and the column to take from it."""
    pattern: str        # glob inside the window folder
    column_key: str     # case-insensitive prefix of the wanted column
    exclude_key: str | None = None  # safety: a column that must NOT be used


# Power columns for windows whose power data comes from raw SMARD files.
SMARD_SOURCES = {
    "Price": SmardSource("*Day-ahead_prices*Hour.csv", "Germany/Luxembourg"),
    "Load_DA_Forecast": SmardSource(
        "*Forecasted_consumption*Hour.csv", "grid load", exclude_key="Residual load"),
    # Paper: RES = solar + wind onshore + wind offshore == SMARD "Photovoltaics and wind".
    "Renewables_DA_Forecast": SmardSource(
        "*Forecasted_generation_Day-Ahead*Hour.csv", "Photovoltaics and wind"),
}


@dataclass(frozen=True)
class DailySource:
    """One daily commodity series."""
    pattern: str        # glob inside the window folder
    kind: str           # "investing" (Investing.com export) or "fred"
    value_column: str
    currency: str       # quote currency; "USD" series are converted to EUR


DAILY_SOURCES = {
    "EUA": DailySource("*Carbon Emissions Futures Historical Data.csv", "investing", "Price", "EUR"),
    "API2_Coal": DailySource("*Rotterdam Coal Futures Historical Data.csv", "investing", "Price", "USD"),
    "TTF_Gas": DailySource("*Dutch TTF Natural Gas Futures Historical Data.csv", "investing", "Price", "EUR"),
    "Brent_oil": DailySource("*BRENTEU.csv", "fred", "DCOILBRENTEU", "USD"),
}


@dataclass(frozen=True)
class WindowConfig:
    """One output dataset.

    power_source:
        "reference" -> Price / load / RES copied from DE.csv (used for 2015-20,
                       because the SMARD DE-LU series only starts 2018-10-01).
        "smard"     -> parsed from the SMARD exports in the window folder.
    reference_fallback:
        When a daily commodity has no quote yet (before the first row of the
        raw file), fill those days from DE.csv instead of back-filling.
    """
    name: str
    folder: str
    start: str          # first calendar day (inclusive)
    end: str            # last calendar day (inclusive)
    output: str
    power_source: str
    reference_fallback: bool = True
    lookback_days: int = 7  # daily calendar starts this many days before `start`
    expected_days: int = 2192
    notes: list[str] = field(default_factory=list)


WINDOWS = {
    "2015-20": WindowConfig(
        name="2015-20",
        folder="2015-20",
        start="2015-01-01",
        end="2020-12-31",
        output="DE_dataset_2015-2020.csv",
        power_source="reference",
    ),
    "2020-26": WindowConfig(
        name="2020-26",
        folder="2020-26",
        start="2020-11-19",
        # 2020-11-19 .. 2026-11-19 is 2,192 days (= 52,608 h, same length as the
        # paper's window). Ending on 2026-11-18 would give only 2,191 days.
        end="2026-11-19",
        output="DE_dataset_2020-2026.csv",
        power_source="smard",
    ),
}

# Overlap period used to cross-check the two windows against DE.csv.
OVERLAP = ("2020-11-19", "2020-12-31")
