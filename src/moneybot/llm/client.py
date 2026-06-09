"""The LLM client seam: the single contract every agent depends on.

Only moneybot.llm.anthropic_client implements this against the real SDK; all
other code (and every test) depends on this Protocol, so no test hits the
network. Mirrors the providers' _fetch* seam discipline.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Send one structured request and return the parsed JSON object.

        The returned dict conforms to `schema`. Implementations must raise
        ValueError if the model output cannot be parsed as JSON.
        """
        ...
