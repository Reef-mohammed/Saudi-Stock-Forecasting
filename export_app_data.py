"""
Copy what the app needs from the research outputs in data/ into app_data/ (committed to git).

  app_data/prices.parquet            daily closes of active companies (float32, ~3 MB)
  app_data/companies.csv             company list with history dates
  app_data/scenario_calibration.json scenario widening factors from calibrate_scenarios.py

The app tops these prices up with recent Yahoo data at runtime, so rerun this only after
rebuilding the dataset or recalibrating.

Usage
  python export_app_data.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

SRC, DST = Path("data"), Path("app_data")


def main() -> None:
    DST.mkdir(exist_ok=True)
    companies = pd.read_csv(SRC / "companies.csv")
    companies = companies[companies["active"]]
    prices = pd.read_parquet(SRC / "prices.parquet", columns=["date", "ticker", "close"])
    prices = prices[prices["ticker"].isin(companies["ticker"])].astype({"close": "float32"})
    prices.to_parquet(DST / "prices.parquet", index=False, compression="zstd")
    companies.to_csv(DST / "companies.csv", index=False)
    shutil.copy(SRC / "scenario_calibration.json", DST / "scenario_calibration.json")
    print(f"{len(prices):,} rows, {len(companies)} companies -> {DST}/")


if __name__ == "__main__":
    main()
