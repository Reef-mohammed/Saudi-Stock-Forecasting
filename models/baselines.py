"""Baselines every real model must beat."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster, lognormal_forecast


class Naive(Forecaster):
    """Random walk: the best guess for any future day is today's price."""

    name = "naive"

    def __init__(self, vol_window: int = 252):
        self.vol_window = vol_window

    def fit(self, close: pd.Series) -> "Naive":
        lr = np.log(close).diff().dropna()
        self.last = float(close.iloc[-1])
        self.sigma = float(lr.tail(self.vol_window).std())
        return self

    def forecast(self, h: int, level: float = 0.8) -> pd.DataFrame:
        return lognormal_forecast(self.last, 0.0, self.sigma, h, level)


class Drift(Naive):
    """Random walk with drift: extends the stock's long-run average log return."""

    name = "drift"

    def __init__(self, vol_window: int = 252, drift_window: int = 5 * 252):
        super().__init__(vol_window)
        self.drift_window = drift_window

    def fit(self, close: pd.Series) -> "Drift":
        super().fit(close)
        lr = np.log(close).diff().dropna()
        self.mu = float(lr.tail(self.drift_window).mean())
        return self

    def forecast(self, h: int, level: float = 0.8) -> pd.DataFrame:
        return lognormal_forecast(self.last, self.mu, self.sigma, h, level)
