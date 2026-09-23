"""
Monte Carlo scenarios for long horizons (1-5 years): worst / likely / best.

Future paths are built by replaying random one-month blocks of the stock's own history (a block
bootstrap). Blocks keep real crashes, fat tails and volatile spells, which a normal distribution
misses. Returns are demeaned first, so the drift is a separate, explicit choice:
  zero    likely path = today's price (the naive model, which won the backtest)
  market  likely path grows at the market's long-run average daily log return
`scale_lo` / `scale_hi` widen the worst / best side separately (crashes break the worst side, booms the
best side); calibrate_scenarios.py sets them from backtests so the band holds even in bad years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BLOCK = 21                        # trading days per block (~1 month)
SCENARIOS = {"worst": 0.1, "likely": 0.5, "best": 0.9}


class MonteCarlo:
    name = "montecarlo"

    def __init__(self, drift: str = "zero", scale_lo=1.0, scale_hi=1.0, n_paths: int = 2000, seed: int = 0):
        """scale_lo / scale_hi: a number, or one value per month (see scale_curve)."""
        self.drift, self.n_paths, self.seed = drift, n_paths, seed
        self.scale_lo, self.scale_hi = scale_lo, scale_hi

    def fit(self, close: pd.Series, market_mu: float = 0.0) -> "MonteCarlo":
        lr = np.log(close.to_numpy(dtype=float))
        lr = np.diff(lr)
        self.mu = {"zero": 0.0, "market": market_mu}[self.drift]
        c = np.concatenate([[0.0], np.cumsum(lr - lr.mean())])
        self.blocks = c[BLOCK:] - c[:-BLOCK]          # every overlapping 1-month return
        self.last = float(close.iloc[-1])
        return self

    def spread_quantiles(self, months: int, quantiles=tuple(SCENARIOS.values())) -> np.ndarray:
        """Quantiles of the simulated cumulative log-return spread, shape (len(quantiles), months)."""
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(self.blocks), size=(self.n_paths, months))
        paths = np.cumsum(self.blocks[idx], axis=1)
        return np.quantile(paths, quantiles, axis=0)

    def center(self, months: int) -> np.ndarray:
        return self.mu * BLOCK * np.arange(1, months + 1)

    def scenarios(self, months: int = 60) -> pd.DataFrame:
        """Monthly worst / likely / best prices for the next `months` months."""
        q = self.spread_quantiles(months)
        c = self.center(months)
        lo, hi = np.broadcast_to(self.scale_lo, months), np.broadcast_to(self.scale_hi, months)
        # Uncertainty never shrinks with time: keep each side at least as wide as the month before
        down = np.maximum.accumulate(np.maximum(-lo * q[0], 0))
        up = np.maximum.accumulate(np.maximum(hi * q[2], 0))
        return pd.DataFrame({"month": np.arange(1, months + 1),
                             "worst": self.last * np.exp(c - down),
                             "likely": self.last * np.exp(c + q[1]),
                             "best": self.last * np.exp(c + up)})


def scale_curve(calib: dict, side: str, months: int) -> np.ndarray:
    """Per-month widening factor from calibrated horizons (flat before 1y, linear in between)."""
    pts = sorted((calib["months"][h], v) for h, v in calib[f"scale_{side}"].items())
    x, y = zip(*pts)
    return np.interp(np.arange(1, months + 1), x, y)
