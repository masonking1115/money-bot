"""The kill switch: a single, always-checkable flag that halts all trading.

Active when either the MONEYBOT_KILL_SWITCH env var is truthy or the configured
file exists on disk. Kept independent of the rest of the engine so it can be
tripped by an operator out-of-band (touch a file) without code or config change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moneybot.config import Settings

_TRUTHY = {"1", "true", "yes", "on"}


def kill_switch_active(settings: Settings) -> bool:
    if os.environ.get("MONEYBOT_KILL_SWITCH", "").strip().lower() in _TRUTHY:
        return True
    return Path(settings.kill_switch_file).exists()
