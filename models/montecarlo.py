"""
Monte Carlo scenarios for long horizons (1-10 years): worst / likely / best.

Future paths are built by replaying random one-month blocks of the stock's own history (a block
bootstrap). Blocks keep real crashes, fat tails and volatile spells, which a normal distribution
misses. Returns are demeaned first, so the drift is a separate, explicit choice:
  zero    likely path = today's price (the naive model, which won the backtest)
  market  likely path grows at the market's long-run average daily log return
`scale` widens or narrows the spread around the drift; calibrate_scenarios.py sets it from backtests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BLOCK = 21                        # trading days per block (~1 month)
SCENARIOS = {"worst": 0.1, "likely": 0.5, "best": 0.9}


class MonteCarlo:
    name = "montecarlo"

    def __init__(self, drift: str = "zero", scale: float = 1.0, n_paths: int = 2000, seed: int = 0):
        self.drift, self.scale, self.n_paths, self.seed = drift, scale, n_paths, seed

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

    def scenarios(self, months: int = 120) -> pd.DataFrame:
        """Monthly worst / likely / best prices for the next `months` months."""
        q = self.spread_quantiles(months)
        c = self.center(months)
        out = {name: self.last * np.exp(c + self.scale * q[i]) for i, name in enumerate(SCENARIOS)}
        return pd.DataFrame({"month": np.arange(1, months + 1), **out})
