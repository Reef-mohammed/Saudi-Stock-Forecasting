"""
Build a clean Saudi (Tadawul) daily price dataset.

Sources
  1. Kaggle "Saudi Stock Exchange (Tadawul)" (2000 -> Mar 2020) for deep history
  2. yfinance (.SR tickers) for 2020 -> today, refreshed on every run

Output (in --out, default ./data)
  prices.parquet (or prices.csv)  long format: date, ticker, open, high, low, close, volume, source
  companies.csv                   ticker, name, sector, first_date, last_date, n_days, short_history
  macro.parquet (or macro.csv)    TASI index + Brent oil closes (useful model features)

Usage
  python build_dataset.py --kaggle data/raw/Tadawul_stcks.csv
  python build_dataset.py --kaggle data/raw/Tadawul_stcks.csv --extra-tickers tickers_extra.csv
  python build_dataset.py --yf-only 2222.SR 1120.SR 7010.SR     # skip Kaggle entirely
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("tadawul")

YF_START = "2019-06-01"          # start yfinance a bit before Kaggle ends -> overlap for checks
MACRO_TICKERS = {"^TASI.SR": "tasi", "BZ=F": "brent"}
SHORT_HISTORY_DAYS = 3 * 252     # < ~3 trading years -> warn in the app
SCALE_TOLERANCE = 0.02           # >2% median price mismatch at overlap -> rescale Kaggle history

# Kaggle column names vary between versions/datasets, so map any known alias -> our name.
COLUMN_ALIASES = {
    "symbol": "symbol", "ticker": "symbol", "code": "symbol",
    "name": "name", "company": "name", "company_name": "name",
    "sectoer": "sector", "sector": "sector",          # 'sectoer' is a typo in the original dataset
    "date": "date",
    "open": "open", "high": "high", "low": "low", "close": "close",
    "volume_traded": "volume", "volume_traded_": "volume", "volume": "volume",
}
PRICE_COLS = ["open", "high", "low", "close"]


# ----------------------------------------------------------------------------- Kaggle
def load_kaggle(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prices, companies) from the Kaggle CSV."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    raw = raw.rename(columns={c: COLUMN_ALIASES[c] for c in raw.columns if c in COLUMN_ALIASES})

    missing = {"symbol", "date", "close"} - set(raw.columns)
    if missing:
        raise ValueError(f"Kaggle file is missing columns {missing}. Found: {list(raw.columns)}")

    raw["ticker"] = raw["symbol"].astype(str).str.extract(r"(\d+)")[0] + ".SR"
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    for c in PRICE_COLS + ["volume"]:
        if c not in raw:
            raw[c] = np.nan
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    prices = raw[["date", "ticker", *PRICE_COLS, "volume"]].copy()
    prices["source"] = "kaggle"

    companies = (
        raw.sort_values("date")
        .groupby("ticker")
        .agg(name=("name", "last") if "name" in raw else ("ticker", "last"),
             sector=("sector", "last") if "sector" in raw else ("ticker", "size"))
        .reset_index()
    )
    if "sector" not in raw:
        companies["sector"] = "Unknown"
    log.info("Kaggle: %d rows, %d tickers", len(prices), prices["ticker"].nunique())
    return prices, companies


# ----------------------------------------------------------------------------- yfinance
def fetch_yfinance(tickers: list[str], start: str, batch_size: int = 40) -> pd.DataFrame:
    """Download daily OHLCV for tickers in batches. Unadjusted prices, to match Kaggle."""
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        log.info("yfinance: batch %d-%d of %d", i + 1, i + len(batch), len(tickers))
        df = yf.download(batch, start=start, auto_adjust=False, group_by="ticker",
                         progress=False, threads=True)
        if df.empty:
            continue
        if not isinstance(df.columns, pd.MultiIndex):          # single ticker
            df.columns = pd.MultiIndex.from_product([batch, df.columns])
        long = df.stack(level=0, future_stack=True).reset_index()
        long.columns = [str(c).lower().replace(" ", "_") for c in long.columns]
        long = long.rename(columns={"level_1": "ticker", "price": "ticker"})
        frames.append(long)
        time.sleep(1)                                          # be polite to the API

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", *PRICE_COLS, "volume", "source"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out[["date", "ticker", *PRICE_COLS, "volume"]]
    out["source"] = "yfinance"
    log.info("yfinance: %d rows, %d tickers", len(out), out["ticker"].nunique())
    return out


def fetch_company_info(tickers: list[str]) -> pd.DataFrame:
    """Names/sectors from yfinance for tickers not in Kaggle (slow: one request each)."""
    import yfinance as yf

    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            rows.append({"ticker": t, "name": info.get("longName") or info.get("shortName") or t,
                         "sector": info.get("sector") or "Unknown"})
        except Exception as e:                                  # network hiccups, delisted, etc.
            log.warning("info failed for %s: %s", t, e)
            rows.append({"ticker": t, "name": t, "sector": "Unknown"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- cleaning & merge
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    # Fill missing open/high/low with close rather than dropping the day
    for c in ["open", "high", "low"]:
        df[c] = df[c].where(df[c] > 0, df["close"])
    # Kaggle sometimes has one-day spikes from bad ticks: drop >60% daily moves that fully revert
    df = df.sort_values(["ticker", "date"])
    r = df.groupby("ticker")["close"].pct_change()
    r_next = df.groupby("ticker")["close"].pct_change(-1)
    spike = (r.abs() > 0.6) & (r_next.abs() > 0.35) & (np.sign(r) == np.sign(r_next))
    if spike.any():
        log.info("Dropping %d one-day price spikes", int(spike.sum()))
    df = df[~spike]
    return df.drop_duplicates(["ticker", "date"], keep="last")


def merge_sources(kaggle: pd.DataFrame, yf_df: pd.DataFrame) -> pd.DataFrame:
    """Prefer yfinance where it exists; rescale Kaggle history if the two disagree at the overlap."""
    if kaggle.empty:
        return yf_df
    parts = []
    for ticker, k in kaggle.groupby("ticker"):
        y = yf_df[yf_df["ticker"] == ticker]
        if y.empty:
            parts.append(k)
            continue
        overlap = k.merge(y[["date", "close"]], on="date", suffixes=("_k", "_y"))
        if len(overlap) >= 5:
            ratio = float(np.median(overlap["close_y"] / overlap["close_k"]))
            if abs(ratio - 1) > SCALE_TOLERANCE:
                log.warning("%s: sources differ by %.1f%% at overlap -> rescaling Kaggle history",
                            ticker, (ratio - 1) * 100)
                k = k.copy()
                k[PRICE_COLS] = k[PRICE_COLS] * ratio
                k["volume"] = k["volume"] / ratio
        parts.append(k[k["date"] < y["date"].min()])
    parts.append(yf_df)
    return pd.concat(parts, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def build_companies(prices: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    stats = prices.groupby("ticker").agg(first_date=("date", "min"), last_date=("date", "max"),
                                         n_days=("date", "size")).reset_index()
    out = stats.merge(known, on="ticker", how="left")
    out["name"] = out["name"].fillna(out["ticker"])
    out["sector"] = out["sector"].fillna("Unknown")
    out["short_history"] = out["n_days"] < SHORT_HISTORY_DAYS
    # Flag tickers that stopped trading (likely delisted/merged) -> hide from the app selector
    out["active"] = out["last_date"] >= prices["date"].max() - pd.Timedelta(days=30)
    return out[["ticker", "name", "sector", "first_date", "last_date", "n_days",
                "short_history", "active"]]


def save(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except ImportError:                                         # no pyarrow -> CSV fallback
        df.to_csv(path.with_suffix(".csv"), index=False)
        return path.with_suffix(".csv")


# ----------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kaggle", type=Path, help="Path to the Kaggle Tadawul CSV")
    p.add_argument("--extra-tickers", type=Path,
                   help="CSV with a 'ticker' column for listings newer than the Kaggle data (e.g. 2222.SR)")
    p.add_argument("--yf-only", nargs="*", help="Skip Kaggle; fetch only these tickers from yfinance")
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    kaggle_prices, known = pd.DataFrame(), pd.DataFrame(columns=["ticker", "name", "sector"])
    tickers: list[str] = list(args.yf_only or [])

    if args.kaggle and not args.yf_only:
        kaggle_prices, known = load_kaggle(args.kaggle)
        kaggle_prices = clean(kaggle_prices)
        tickers = sorted(kaggle_prices["ticker"].unique())
    if args.extra_tickers:
        extra = pd.read_csv(args.extra_tickers)["ticker"].astype(str).str.strip().tolist()
        tickers = sorted(set(tickers) | set(extra))
    if not tickers:
        p.error("Give --kaggle, --extra-tickers, or --yf-only")

    yf_prices = clean(fetch_yfinance(tickers, YF_START))
    prices = merge_sources(kaggle_prices, yf_prices)

    unknown = sorted(set(prices["ticker"]) - set(known["ticker"]))
    if unknown:
        log.info("Fetching names/sectors for %d tickers not in Kaggle", len(unknown))
        known = pd.concat([known, fetch_company_info(unknown)], ignore_index=True)

    companies = build_companies(prices, known)

    macro = fetch_yfinance(list(MACRO_TICKERS), "2000-01-01")
    macro = (macro.pivot_table(index="date", columns="ticker", values="close")
             .rename(columns=MACRO_TICKERS).reset_index())

    for name, df in [("prices", prices), ("macro", macro)]:
        log.info("Saved %s", save(df, args.out / name))
    companies.to_csv(args.out / "companies.csv", index=False)

    log.info("Done: %d rows | %d tickers (%d active, %d short history) | %s -> %s",
             len(prices), len(companies), companies["active"].sum(), companies["short_history"].sum(),
             prices["date"].min().date(), prices["date"].max().date())


if __name__ == "__main__":
    main()
