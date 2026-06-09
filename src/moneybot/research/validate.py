"""Pure validation + grounding for raw signals emitted by the deep-read tier.

Three guards, in order: (1) coerce the raw dict into a CatalystSignal (Pydantic
rejects bad ranges/enums); (2) bound the ticker to the one we asked about;
(3) ground every citation — drop the whole signal if it has no evidence or cites
any URL we did not provide (anti-hallucination). Surviving signals get a
deterministic, content-addressed signal_id.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from moneybot.strategies.models import CatalystSignal


def _evidence_url(e: Any) -> str:
    """Read a URL from an Evidence model or a raw dict.

    model_copy(update=...) does not re-validate, so signal.evidence may hold
    plain dicts rather than Evidence instances; handle both.
    """
    return e.url if hasattr(e, "url") else e["url"]


def make_signal_id(signal: CatalystSignal) -> str:
    """Content hash over the fields that define a distinct catalyst."""
    urls = "|".join(sorted(_evidence_url(e) for e in signal.evidence))
    payload = f"{signal.ticker}|{signal.category}|{signal.thesis}|{urls}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def validate_signals(
    raw_signals: Iterable[dict[str, Any]],
    *,
    ticker: str,
    allowed_urls: set[str],
) -> list[CatalystSignal]:
    """Return only the well-formed, on-ticker, fully-grounded signals."""
    valid: list[CatalystSignal] = []
    for raw in raw_signals:
        try:
            signal = CatalystSignal.model_validate(raw)
        except ValidationError:
            continue  # malformed -> skip, never raise on model output

        if signal.ticker != ticker:
            continue
        if not signal.evidence:
            continue
        if any(e.url not in allowed_urls for e in signal.evidence):
            continue  # any ungrounded citation invalidates the signal

        valid.append(signal.model_copy(update={"signal_id": make_signal_id(signal)}))
    return valid
