"""Shared interface every forecasting model implements, so backtest.py and the app can swap them freely."""

from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import NormalDist

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """Fit on a daily close-price series, forecast prices h trading days ahead.

    Models work in log returns internally (prices are non-stationary) but take and return prices.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, close: pd.Series) -> "Forecaster":
        """close: daily close prices indexed by date, oldest first, no NaNs."""

    @abstractmethod
    def forecast(self, h: int, level: float = 0.8) -> pd.DataFrame:
        """Return rows for steps 1..h with columns: step, mean, lo, hi.

        mean is the median price path; lo/hi bound the central `level` interval.
        """


def lognormal_forecast(last: float, mu: float, sigma: float, h: int, level: float) -> pd.DataFrame:
    """Price forecast when log returns are i.i.d. N(mu, sigma^2): used by the baselines."""
    steps = np.arange(1, h + 1)
    z = NormalDist().inv_cdf(0.5 + level / 2)
    center = np.log(last) + mu * steps
    spread = z * sigma * np.sqrt(steps)
    return pd.DataFrame({"step": steps, "mean": np.exp(center),
                         "lo": np.exp(center - spread), "hi": np.exp(center + spread)})
