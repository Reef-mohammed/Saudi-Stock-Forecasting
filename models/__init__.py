from .base import Forecaster
from .baselines import Drift, Naive

MODELS: dict[str, type[Forecaster]] = {m.name: m for m in (Naive, Drift)}
