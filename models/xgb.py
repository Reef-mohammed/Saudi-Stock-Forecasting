"""
Global XGBoost model: one model learns from every ticker at once (far more data than any single stock).

For each horizon it predicts the 10th/50th/90th percentile of the forward log return with quantile
regression, so it produces a forecast and an 80% interval directly, like the other models.

Unlike the per-series models it is trained on the whole panel, so it has its own fit/predict methods
instead of the Forecaster interface; backtest.py handles it separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = [0.1, 0.5, 0.9]
TRAIN_EVERY = 5          # use every 5th day for training: forward returns overlap, so little is lost


def market_index(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight market daily log return across all tickers (a TASI proxy back to 2001)."""
    lr = np.log(prices["close"]).groupby(prices["ticker"]).diff()
    return lr.groupby(prices["date"]).mean().rename("mkt_lr")


def build_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Features at each (ticker, date), using only information available at that day's close."""
    df = prices[["date", "ticker", "close"]].sort_values(["ticker", "date"]).reset_index(drop=True)
    df["date"] = df["date"].astype("datetime64[ns]")            # parquet files may store ms or us
    macro = macro.assign(date=macro["date"].astype("datetime64[ns]"))
    logc = np.log(df["close"])
    g = logc.groupby(df["ticker"])
    lr = g.diff()
    by = lr.groupby(df["ticker"])

    for n in (1, 5, 21, 63, 126, 252):
        df[f"r{n}"] = g.diff(n)
    df["vol21"] = by.transform(lambda s: s.rolling(21).std())
    df["vol63"] = by.transform(lambda s: s.rolling(63).std())
    df["vol_ratio"] = df["vol21"] / df["vol63"]
    df["dist_high252"] = logc - g.transform(lambda s: s.rolling(252, min_periods=63).max())
    df["dist_ma200"] = logc - g.transform(lambda s: s.rolling(200, min_periods=100).mean())

    mkt = market_index(df).sort_index()
    mkt_df = pd.DataFrame({"date": mkt.index,
                           "mkt_r21": mkt.rolling(21).sum().values,
                           "mkt_r63": mkt.rolling(63).sum().values,
                           "mkt_vol21": mkt.rolling(21).std().values})
    df = df.merge(mkt_df, on="date", how="left")
    df["rel_r63"] = df["r63"] - df["mkt_r63"]

    brent = macro[["date", "brent"]].dropna().sort_values("date")
    lb = np.log(brent["brent"])
    brent = brent.assign(brent_r21=lb.diff(21), brent_r63=lb.diff(63))[["date", "brent_r21", "brent_r63"]]
    # Brent settles after Tadawul closes, so a same-date Brent price would leak the future: use the day before
    df = pd.merge_asof(df.sort_values("date"), brent, on="date", allow_exact_matches=False)
    # Suspended stocks have zero volatility, so ratios can be infinite: treat as missing
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


FEATURES = ["r1", "r5", "r21", "r63", "r126", "r252", "vol21", "vol63", "vol_ratio",
            "dist_high252", "dist_ma200", "mkt_r21", "mkt_r63", "mkt_vol21", "rel_r63",
            "brent_r21", "brent_r63"]


def add_targets(feats: pd.DataFrame, horizons: dict[str, int]) -> pd.DataFrame:
    """Forward log return per horizon, plus the date it becomes known (to avoid training on the future)."""
    g_close = np.log(feats["close"]).groupby(feats["ticker"])
    g_date = feats["date"].groupby(feats["ticker"])
    for label, h in horizons.items():
        feats[f"y_{label}"] = g_close.shift(-h) - np.log(feats["close"])
        feats[f"known_{label}"] = g_date.shift(-h)
    return feats


class GlobalXGB:
    name = "xgb"

    def __init__(self, horizons: dict[str, int], n_estimators: int = 300):
        self.horizons = horizons
        self.n_estimators = n_estimators
        self.models: dict[str, object] = {}

    def fit(self, panel: pd.DataFrame, cutoff: pd.Timestamp) -> "GlobalXGB":
        """Train on rows whose target was already known on `cutoff` (strictly no look-ahead)."""
        import xgboost as xgb

        sample = panel[panel.groupby("ticker").cumcount() % TRAIN_EVERY == 0]
        for label in self.horizons:
            rows = sample[sample[f"known_{label}"] <= cutoff].dropna(subset=[f"y_{label}", "r252"])
            params = {"objective": "reg:quantileerror", "quantile_alpha": np.array(QUANTILES),
                      "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8,
                      "colsample_bytree": 0.8, "min_child_weight": 50, "tree_method": "hist"}
            dtrain = xgb.DMatrix(rows[FEATURES], label=rows[f"y_{label}"])
            self.models[label] = xgb.train(params, dtrain, num_boost_round=self.n_estimators)
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """rows: panel rows at forecast origins. Returns one row per (input row, horizon) with prices."""
        import xgboost as xgb

        out = []
        dm = xgb.DMatrix(rows[FEATURES])
        for label, h in self.horizons.items():
            q = np.sort(self.models[label].predict(dm), axis=1)              # keep quantiles ordered
            close = rows["close"].to_numpy()
            out.append(pd.DataFrame({
                "ticker": rows["ticker"].to_numpy(), "origin": rows["date"].to_numpy(),
                "horizon": label, "h": h, "origin_price": close,
                "lo": close * np.exp(q[:, 0]), "pred": close * np.exp(q[:, 1]),
                "hi": close * np.exp(q[:, 2])}))
        return pd.concat(out, ignore_index=True)
