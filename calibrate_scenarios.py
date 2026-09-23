"""
Backtest and calibrate the Monte Carlo scenarios at 1, 3 and 5 years.

For each ticker and origin (every quarter from 2005) the simulator sees only past prices. We record
where the real outcome landed relative to the simulated worst/best band, then widen each side:
  worst  at most 1 in 10 outcomes below it in EVERY start year, including the 2006 crash. Averages
         hide crashes, because a market-wide crash makes almost every stock miss at once, and a
         too-optimistic worst case is the mistake that costs an investor money.
  best   at most 1 in 10 outcomes above it on average. Beating the best case is a pleasant
         surprise, and sizing it for the 2005 boom would make it absurdly optimistic.
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
MISS_PER_SIDE = 0.10              # "worst" / "best" should each be beaten by about 1 in 10 outcomes


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


def prepare(res: pd.DataFrame) -> pd.DataFrame:
    dev = res["actual"] - res["center"]
    # How far out the actual landed, in units of the band's half-width on its side: the band widened
    # by k on that side contains the actual exactly when ratio <= k.
    return res.assign(down=dev < 0, year=res["origin"].dt.year,
                      ratio=np.where(dev >= 0, dev / res["q90"], dev / res["q10"]),
                      ape=(np.exp(res["center"] + res["q50"] - res["actual"]) - 1).abs())


def side_scale(rows: pd.DataFrame, down: bool) -> float:
    """Smallest widening so at most MISS_PER_SIDE of these forecasts land beyond that side."""
    side = np.sort(rows.loc[rows["down"] == down, "ratio"].to_numpy())
    allowed = int(np.floor(MISS_PER_SIDE * len(rows)))
    return 0.0 if len(side) <= allowed else float(side[len(side) - allowed - 1])


def calibrate(res: pd.DataFrame) -> pd.DataFrame:
    """Per horizon: worst side sized for the worst start year, best side for the average."""
    out = []
    for h, x in res.groupby("horizon"):
        lo_by_year = x.groupby("year").apply(side_scale, True, include_groups=False)
        k_lo, k_hi = lo_by_year.max(), side_scale(x, False)
        below = (x["down"] & (x["ratio"] > k_lo)).groupby(x["year"]).mean()
        above = (~x["down"] & (x["ratio"] > k_hi)).groupby(x["year"]).mean()
        out.append({"horizon": h, "n": len(x), "likely_mape": x["ape"].mean(),
                    "raw_coverage": (x["ratio"] <= 1).mean(),
                    "scale_lo": k_lo, "lo_set_by": int(lo_by_year.idxmax()), "scale_hi": k_hi,
                    "coverage": 1 - below.mean() - above.mean(),
                    "below_worst": float((x["down"] & (x["ratio"] > k_lo)).mean()),
                    "below_worst_max_year": below.max(),
                    "above_best": float((~x["down"] & (x["ratio"] > k_hi)).mean()),
                    "above_best_max_year": above.max(), "above_best_max_year_is": int(above.idxmax()),
                    "origins": f"{x['year'].min()}-{x['year'].max()}"})
    out = pd.DataFrame(out)
    return out.sort_values("horizon", key=lambda s: s.map(HORIZON_MONTHS)).reset_index(drop=True)


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
    res = prepare(res)

    # Pick the drift whose likely line was more accurate on average across horizons
    mape = res.groupby(["drift", "horizon"])["ape"].mean()
    print("Likely-line error (mean APE):\n" + mape.unstack().round(3).to_string())
    drift = mape.groupby("drift").mean().idxmin()

    summary = calibrate(res[res["drift"] == drift]).set_index("horizon")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(f"\nCalibration (drift={drift}):\n" + summary.to_string())

    calib = {"drift": drift, "months": HORIZON_MONTHS, "miss_per_side": MISS_PER_SIDE,
             "scale_lo": summary["scale_lo"].to_dict(), "scale_hi": summary["scale_hi"].to_dict(),
             "tested": {h: {"n": int(s["n"]), "origins": s["origins"],
                            "coverage": round(float(s["coverage"]), 3),
                            "below_worst": round(float(s["below_worst"]), 3),
                            "below_worst_max_year": round(float(s["below_worst_max_year"]), 3),
                            "above_best": round(float(s["above_best"]), 3)}
                        for h, s in summary.iterrows()}}
    (args.data / "scenario_calibration.json").write_text(json.dumps(calib, indent=2))
    log.info("Saved %s", args.data / "scenario_calibration.json")


if __name__ == "__main__":
    main()
