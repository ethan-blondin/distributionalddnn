"""Build the DDNN input datasets for the 2015-20 and 2020-26 windows.

Usage (from the repository root, i.e. the parent of Datasets/):
    python build_datasets.py                 # build both windows + validate
    python build_datasets.py --window 2020-26
    python build_datasets.py --data-dir path/to/Datasets --no-validate

Outputs (schema and column order of Datasets/DE.csv, 24 rows per day):
    Datasets/DE_dataset_2015-2020.csv
    Datasets/DE_dataset_2020-2026.csv
See data_pipeline/ for the processing rules.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline import (WINDOWS, alignment_table, build_window, load_fx, load_reference,
                           report_frame, validate_outputs, write_dataset)
from data_pipeline.config import DATA_DIR


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--window", choices=list(WINDOWS), action="append",
                    help="window(s) to build (default: all)")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)

    reference = load_reference(args.data_dir)
    fx = load_fx(args.data_dir)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    for name in args.window or list(WINDOWS):
        result = build_window(WINDOWS[name], args.data_dir, reference=reference, fx=fx)
        path = write_dataset(result, args.data_dir)
        print(f"\n### {name}: wrote {path} ({len(result.data):,} rows)")
        print(report_frame(result).to_string())
        print("\nDaily alignment vs DE.csv (best_lag_days = 0 -> no lag baked in):")
        print(alignment_table(reference, result.daily_eur, result.daily_has_quote).to_string(index=False))

    if not args.no_validate:
        print()
        validate_outputs(args.data_dir)


if __name__ == "__main__":
    main()
