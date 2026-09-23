"""
Walk-forward (expanding window) backtest of every model on every active ticker.

At each forecast origin a model sees only prices up to that day, forecasts ahead, and is scored
against what actually happened. Origins step forward through time, so no future data ever leaks in.

Output (in data/)
  backtest_results.parquet   one row per ticker x origin x model x horizon
  backtest_metrics.csv       summary per model x horizon (the numbers to show in the app / on LinkedIn)

Usage
  python backtest.py
  python backtest.py --models naive drift --tickers 2222.SR 1120.SR --step 63
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models import MODELS

log = logging.getLogger("backtest")

HORIZONS = {"1w": 5, "1m": 21, "3m": 63, "1y": 252}   # trading days
MIN_TRAIN = 3 * 252                                   # need 3 years of history before the first origin
LEVEL = 0.8                                           # interval width scored for coverage


def origin_positions(close: pd.Series, start: pd.Timestamp, step: int) -> range:
    first = max(MIN_TRAIN, int(close.index.searchsorted(start)))
    return range(first, len(close) - min(HORIZONS.values()), step)


def backtest_ticker(close: pd.Series, model_names: list[str], start: pd.Timestamp,
                    step: int) -> list[dict]:
    rows = []
    h_max = max(HORIZONS.values())
    for o in origin_positions(close, start, step):
        train = close.iloc[:o + 1]
        origin_price = float(train.iloc[-1])
        for name in model_names:
            fc = MODELS[name]().fit(train).forecast(h_max, LEVEL).set_index("step")
            for label, h in HORIZONS.items():
                if o + h >= len(close):
                    continue
                actual = float(close.iloc[o + h])
                f = fc.loc[h]
                rows.append({"origin": close.index[o], "model": name, "horizon": label, "h": h,
                             "origin_price": origin_price, "actual": actual, "pred": f["mean"],
                             "lo": f["lo"], "hi": f["hi"]})
    return rows


def backtest_xgb(prices: pd.DataFrame, macro: pd.DataFrame, tickers: list[str],
                 start: pd.Timestamp, step: int) -> pd.DataFrame:
    """Global model: retrain once a year on all tickers, forecast every origin in that year."""
    from models.xgb import GlobalXGB, add_targets, build_features

    panel = add_targets(build_features(prices[prices["ticker"].isin(tickers)], macro), HORIZONS)
    # The same origins the per-series models are scored on
    is_origin = np.zeros(len(panel), dtype=bool)
    for t, idx in panel.groupby("ticker").indices.items():
        is_origin[idx[list(origin_positions(pd.Series(index=panel["date"].iloc[idx]), start, step))]] = True
    origins = panel[is_origin]

    preds = []
    for year in sorted(origins["date"].dt.year.unique()):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        rows = origins[origins["date"].dt.year == year]
        log.info("xgb: training with data known by %s, forecasting %d origins", cutoff.date(), len(rows))
        preds.append(GlobalXGB(HORIZONS).fit(panel, cutoff).predict(rows))
    res = pd.concat(preds, ignore_index=True)

    actual = pd.concat([origins[["ticker", "date"]].assign(horizon=label,
                        actual=origins["close"] * np.exp(origins[f"y_{label}"])) for label in HORIZONS])
    res = res.merge(actual.rename(columns={"date": "origin"}), on=["ticker", "origin", "horizon"])
    return res.dropna(subset=["actual"]).assign(model="xgb")


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    res = res.assign(
        ape=(res["pred"] / res["actual"] - 1).abs(),
        log_err=np.log(res["pred"] / res["actual"]),
        covered=(res["actual"] >= res["lo"]) & (res["actual"] <= res["hi"]),
        pred_move=np.sign(res["pred"] - res["origin_price"])
                  * ~np.isclose(res["pred"], res["origin_price"], rtol=1e-9),
        real_move=np.sign(res["actual"] - res["origin_price"]),
    )
    g = res.groupby(["model", "horizon", "h"])
    out = g.agg(n=("ape", "size"), mape=("ape", "mean"), median_ape=("ape", "median"),
                rmse_log=("log_err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                coverage=("covered", "mean")).reset_index()
    # Direction hit rate only counts forecasts that call a direction (the naive model never does)
    moved = res[res["pred_move"] != 0]
    hits = (moved["pred_move"] == moved["real_move"]).groupby([moved["model"], moved["horizon"]]).mean()
    out["direction_hit"] = [hits.get((m, hz), np.nan) for m, hz in zip(out["model"], out["horizon"])]
    # Skill vs the naive random walk: < 1 means better than "tomorrow = today"
    naive = out[out["model"] == "naive"].set_index("horizon")["mape"]
    out["mape_vs_naive"] = out["mape"] / out["horizon"].map(naive)
    return out.sort_values(["h", "model"]).drop(columns="h").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--tickers", nargs="*", help="Default: all active tickers")
    p.add_argument("--start", default="2012-01-01", help="First forecast origin")
    p.add_argument("--step", type=int, default=21, help="Trading days between forecast origins")
    p.add_argument("--no-xgb", dest="xgb", action="store_false", help="Skip the global XGBoost model")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if "naive" not in args.models:
        args.models = ["naive", *args.models]               # needed for the skill score

    prices = pd.read_parquet(args.data / "prices.parquet")
    companies = pd.read_csv(args.data / "companies.csv")
    tickers = args.tickers or companies.loc[companies["active"], "ticker"].tolist()

    rows = []
    for i, t in enumerate(tickers, 1):
        close = prices.loc[prices["ticker"] == t].set_index("date")["close"].sort_index()
        rows += [dict(r, ticker=t) for r in
                 backtest_ticker(close, args.models, pd.Timestamp(args.start), args.step)]
        if i % 25 == 0:
            log.info("%d/%d tickers", i, len(tickers))

    res = pd.DataFrame(rows)
    if args.xgb:
        macro = pd.read_parquet(args.data / "macro.parquet")
        res = pd.concat([res, backtest_xgb(prices, macro, tickers, pd.Timestamp(args.start), args.step)],
                        ignore_index=True)
    res.to_parquet(args.data / "backtest_results.parquet", index=False)
    metrics = summarize(res)
    metrics.to_csv(args.data / "backtest_metrics.csv", index=False)
    log.info("%d forecasts scored across %d tickers", len(res), res["ticker"].nunique())
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
