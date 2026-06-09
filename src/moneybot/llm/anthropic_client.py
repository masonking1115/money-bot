"""The only module that talks to the Anthropic API.

The raw SDK call is isolated in `_create_message` so tests override it and no
test hits the network. `complete_json` builds the request (structured-output
config + tiered thinking), calls the seam, and parses the JSON response.
"""

from __future__ import annotations

import json
from typing import Any

from moneybot.llm.client import LLMClient


class AnthropicClient(LLMClient):
    def __init__(self, client: Any | None = None, *, max_tokens: int = 4096) -> None:
        if client is None:
            from anthropic import Anthropic  # imported lazily so tests need no key

            client = Anthropic()
        self._client = client
        self._max_tokens = max_tokens

    def _create_message(self, **kwargs: Any) -> Any:
        """Network seam — the only line that calls the SDK. Overridden in tests."""
        return self._client.messages.create(**kwargs)

    def complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            # Structured output: forces JSON conforming to `schema`. The installed
            # anthropic SDK (>=0.40; verified on 0.109) takes the schema under
            # output_config.format with type "json_schema" — the deprecated
            # top-level output_format is not used. JSONOutputFormatParam has no
            # "name" field, so only type + schema are passed.
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        # Haiku supports neither adaptive thinking nor effort; only enable it for
        # the deeper tiers (sonnet/opus).
        if not model.startswith("claude-haiku"):
            kwargs["thinking"] = {"type": "adaptive"}

        message = self._create_message(**kwargs)
        # Adaptive thinking can place a thinking block before the text block, so
        # scan for the first block carrying text rather than assuming content[0].
        text = next(
            (t for b in message.content if (t := getattr(b, "text", None)) is not None),
            None,
        )
        if text is None:
            kinds = [type(b).__name__ for b in message.content]
            raise ValueError(f"no text content block in model response (blocks: {kinds})")
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"could not parse JSON from model output: {text!r}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
        return parsed
