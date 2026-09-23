"""
Backtest and calibrate the Monte Carlo scenarios at 1, 3 and 5 years.

For each ticker and origin (every quarter from 2005) the simulator sees only past prices. We record
where the real outcome landed relative to the simulated worst/best band, then pick the widening
factor per horizon that makes the worst-best band contain the real outcome 80% of the time.
The app stops at 5 years: 10-year outcomes exist only for 2005-2016 origins, which overlap so heavily
that they cover one or two independent periods, too few to calibrate on.

Output: data/scenario_calibration.json (read by the app) and a summary table.

Usage
  python calibrate_scenarios.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models.montecarlo import BLOCK, MonteCarlo
from models.xgb import market_index

log = logging.getLogger("calibrate")

HORIZON_MONTHS = {"1y": 12, "3y": 36, "5y": 60}
MIN_TRAIN = 3 * 252
TARGET = 0.8


def run(prices: pd.DataFrame, tickers: list[str], start: str, step: int, n_paths: int) -> pd.DataFrame:
    prices = prices.assign(date=prices["date"].astype("datetime64[ns]"))
    mkt_mu = market_index(prices).sort_index().expanding().mean()     # long-run market drift, past only
    max_m = max(HORIZON_MONTHS.values())
    rows = []
    for i, t in enumerate(tickers, 1):
        close = prices.loc[prices["ticker"] == t].set_index("date")["close"].sort_index()
        first = max(MIN_TRAIN, int(close.index.searchsorted(pd.Timestamp(start))))
        for o in range(first, len(close) - HORIZON_MONTHS["1y"] * BLOCK, step):
            train = close.iloc[:o + 1]
            for drift in ("zero", "market"):
                mc = MonteCarlo(drift, n_paths=n_paths, seed=o).fit(train, float(mkt_mu.asof(train.index[-1])))
                q = mc.spread_quantiles(max_m)
                c = mc.center(max_m)
                for label, m in HORIZON_MONTHS.items():
                    if o + m * BLOCK >= len(close):
                        continue
                    rows.append({"ticker": t, "origin": close.index[o], "drift": drift, "horizon": label,
                                 "actual": np.log(close.iloc[o + m * BLOCK] / close.iloc[o]),
                                 "center": c[m - 1], "q10": q[0, m - 1], "q50": q[1, m - 1],
                                 "q90": q[2, m - 1]})
        if i % 25 == 0:
            log.info("%d/%d tickers", i, len(tickers))
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    dev = res["actual"] - res["center"]
    # How far out the actual landed, in units of the band's half-width on that side. The band
    # scaled by k contains the actual exactly when ratio <= k, so the 80th percentile is the right k.
    res = res.assign(ratio=np.where(dev >= 0, dev / res["q90"], dev / res["q10"]),
                     ape=(np.exp(res["center"] + res["q50"] - res["actual"]) - 1).abs())
    g = res.groupby(["drift", "horizon"])
    out = g.agg(n=("ratio", "size"), likely_mape=("ape", "mean"), likely_median_ape=("ape", "median"),
                raw_coverage=("ratio", lambda r: float((r <= 1).mean())),
                scale=("ratio", lambda r: float(np.quantile(r, TARGET)))).reset_index()
    out["h"] = out["horizon"].map(HORIZON_MONTHS)
    return out.sort_values(["h", "drift"]).drop(columns="h").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--step", type=int, default=63, help="Trading days between origins")
    p.add_argument("--paths", type=int, default=1000)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    prices = pd.read_parquet(args.data / "prices.parquet")
    companies = pd.read_csv(args.data / "companies.csv")
    tickers = args.tickers or companies.loc[companies["active"], "ticker"].tolist()

    res = run(prices, tickers, args.start, args.step, args.paths)
    res.to_parquet(args.data / "scenario_backtest.parquet", index=False)
    summary = summarize(res)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print(summary.to_string(index=False))

    # Pick the drift whose likely line was more accurate on average across horizons
    drift = summary.groupby("drift")["likely_mape"].mean().idxmin()
    chosen = summary[summary["drift"] == drift].set_index("horizon")
    origins = res[res["drift"] == drift].groupby("horizon")["origin"].agg(["min", "max"])
    calib = {"drift": drift, "target_coverage": TARGET, "scale": chosen["scale"].to_dict(),
             "months": HORIZON_MONTHS,
             "tested": {h: {"n": int(chosen.loc[h, "n"]),
                            "origins": f"{origins.loc[h, 'min']:%Y}-{origins.loc[h, 'max']:%Y}"}
                        for h in HORIZON_MONTHS}}
    (args.data / "scenario_calibration.json").write_text(json.dumps(calib, indent=2))
    log.info("Chose drift=%s, scale=%s", drift, {k: round(v, 2) for k, v in calib["scale"].items()})


if __name__ == "__main__":
    main()
