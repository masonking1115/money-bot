"""Wire a RiskEngine from settings: resolve the active strategy.

The Risk Engine uses no LLM, so unlike the agent factories there is no client to
construct — it just resolves the configured strategy (for its per-name/sector
caps) and hands over the data layer and settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import moneybot.strategies  # noqa: F401  -- import for side-effect: registers strategies
from moneybot.risk.engine import RiskEngine
from moneybot.strategies import registry

if TYPE_CHECKING:
    from moneybot.config import Settings
    from moneybot.data_layer import DataLayer


def build_risk_engine(
    *,
    settings: Settings,
    data_layer: DataLayer,
) -> RiskEngine:
    strategy = registry.get(settings.strategy)
    return RiskEngine(data_layer=data_layer, strategy=strategy, settings=settings)
