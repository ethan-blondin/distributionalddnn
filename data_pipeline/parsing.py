"""Robust readers for the raw file formats (SMARD, Investing.com, FRED, DE.csv).

All readers return plain pandas objects indexed by naive local timestamps
(Europe/Berlin wall-clock time, as in the source files and in DE.csv).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "-", ".", "nan", "NaN", "NA", "N/A", "n/a", "None", "null"}

# Tried in order; the first format that parses *every* non-missing value wins.
DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%b %d, %Y %I:%M %p",   # SMARD English: "Nov 19, 2020 12:00 AM"
    "%d.%m.%Y %H:%M",       # SMARD German:  "19.11.2020 00:00"
    "%d.%m.%Y",
    "%m/%d/%Y",             # Investing.com: "11/19/2020"
    "%b %d, %Y",
]


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def sniff_delimiter(path: Path, candidates: str = ";,\t") -> str:
    """Return the candidate delimiter that occurs most often in the header line."""
    with open(path, encoding="utf-8-sig") as fh:
        header = fh.readline()
    counts = {c: header.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise ValueError(f"Cannot detect delimiter of {path.name}: header={header!r}")
    return best


def read_raw_table(path: Path) -> pd.DataFrame:
    """Read any of the CSV exports as strings (no type guessing, BOM stripped)."""
    return pd.read_csv(
        path,
        sep=sniff_delimiter(path),
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
        skipinitialspace=True,
    )


def _detect_decimal(values: pd.Series) -> str:
    """Vote between English ('1,234.56') and German ('1.234,56') notation.

    Only unambiguous tokens vote; e.g. '1,234' could be either and is ignored.
    Defaults to '.' when nothing is conclusive.
    """
    english = german = 0
    for v in values.unique():
        has_dot, has_comma = "." in v, "," in v
        if has_dot and has_comma:
            if v.rfind(".") > v.rfind(","):
                english += 1
            else:
                german += 1
        elif has_comma:
            if not re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", v):
                german += 1          # e.g. "81,25"
        elif has_dot:
            if not re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+", v):
                english += 1         # e.g. "81.25"
    return "," if german > english else "."


def parse_number_series(raw: pd.Series, name: str = "") -> pd.Series:
    """Convert a string column to float64.

    * strips whitespace / NBSP and a trailing '%'
    * '-', '.', '' and similar tokens -> NaN
    * auto-detects thousands/decimal separators for the whole column
    Raises if any non-missing token still fails to parse (no silent corruption).
    """
    s = raw.astype(str).str.strip().str.replace(" ", "", regex=False)
    s = s.str.replace(" ", "", regex=False).str.rstrip("%")
    s = s.mask(s.isin(MISSING_TOKENS))
    present = s.dropna()
    if present.empty:
        return pd.Series(np.nan, index=raw.index, dtype="float64", name=raw.name)

    if _detect_decimal(present) == ".":
        s = s.str.replace(",", "", regex=False)
    else:
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    out = pd.to_numeric(s, errors="coerce")
    bad = out.isna() & s.notna()
    if bad.any():
        examples = raw[bad].unique()[:5]
        raise ValueError(f"{name or raw.name}: {bad.sum()} unparsable values, e.g. {examples}")
    return out.astype("float64")


def parse_dates(raw: pd.Series, name: str = "") -> pd.Series:
    """Parse a date/time column, trying DATE_FORMATS in order."""
    s = raw.astype(str).str.strip()
    s = s.mask(s.isin(MISSING_TOKENS))
    present = s.dropna()
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(present, format=fmt, errors="coerce")
        if parsed.notna().all():
            return pd.to_datetime(s, format=fmt, errors="coerce")
    raise ValueError(f"{name}: no known date format matches, e.g. {present.iloc[:3].tolist()}")


def find_file(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one file matching {pattern!r} in {folder}, "
                                f"found {[m.name for m in matches]}")
    return matches[0]


def find_column(columns, key: str, exclude: str | None = None) -> str:
    """Pick the column whose name starts with `key` (case-insensitive)."""
    hits = [c for c in columns if c.strip().lower().startswith(key.lower())]
    if exclude:
        hits = [c for c in hits if exclude.lower() not in c.lower()]
    if len(hits) != 1:
        raise KeyError(f"Column starting with {key!r} not uniquely found in {list(columns)}")
    return hits[0]


# --------------------------------------------------------------------------- #
# Source-specific readers
# --------------------------------------------------------------------------- #
def read_smard(path: Path, column_key: str, exclude_key: str | None = None) -> pd.Series:
    """Read one SMARD hourly column.

    Returns a float Series indexed by the *Start date* (naive local time).
    The index is NOT unique on fall-back days (two 02:00 rows) and lacks the
    02:00 row on spring-forward days; see transforms.normalize_hourly().
    """
    df = read_raw_table(path)
    col = find_column(df.columns, column_key, exclude_key)
    start_col = find_column(df.columns, "Start date")
    idx = parse_dates(df[start_col], f"{path.name}:{start_col}")
    values = parse_number_series(df[col], f"{path.name}:{col}")
    out = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(idx), name=col)
    return out[out.index.notna()]


def read_investing(path: Path, value_column: str = "Price") -> pd.Series:
    """Read an Investing.com daily export (quoted CSV, MM/DD/YYYY, newest first)."""
    df = read_raw_table(path)
    dates = parse_dates(df["Date"], f"{path.name}:Date").dt.normalize()
    values = parse_number_series(df[value_column], f"{path.name}:{value_column}")
    s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=path.stem)
    return _clean_daily(s)


def read_fred(path: Path, value_column: str) -> pd.Series:
    """Read a FRED daily export (YYYY-MM-DD, '.' or empty for holidays)."""
    df = read_raw_table(path)
    date_col = df.columns[0]
    dates = parse_dates(df[date_col], f"{path.name}:{date_col}").dt.normalize()
    values = parse_number_series(df[value_column], f"{path.name}:{value_column}")
    s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=path.stem)
    return _clean_daily(s)


def _clean_daily(s: pd.Series) -> pd.Series:
    """Sort ascending, drop missing values and duplicated dates (keep last)."""
    s = s[s.index.notna()].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def read_reference(path: Path) -> pd.DataFrame:
    """Read the golden DE.csv (first column = timestamp)."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.DatetimeIndex(parse_dates(pd.Series(df.index), path.name))
    df.index.name = None
    return df.astype("float64")
