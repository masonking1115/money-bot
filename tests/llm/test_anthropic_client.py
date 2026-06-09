import json

import pytest

from moneybot.llm.anthropic_client import AnthropicClient


class FakeContentBlock:
    def __init__(self, text):
        self.text = text


class FakeMessage:
    def __init__(self, text):
        self.content = [FakeContentBlock(text)]


def test_complete_json_parses_model_text_and_captures_request():
    captured = {}

    class Adapter(AnthropicClient):
        def _create_message(self, **kwargs):
            captured.update(kwargs)
            return FakeMessage(json.dumps({"signals": [{"ticker": "NVDA"}]}))

    client = Adapter(client=object())  # real SDK client never used; seam is overridden
    schema = {"type": "object", "properties": {"signals": {"type": "array"}}}
    out = client.complete_json(
        model="claude-sonnet-4-6",
        system="sys",
        user="usr",
        schema=schema,
    )

    assert out == {"signals": [{"ticker": "NVDA"}]}
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["system"] == "sys"
    assert captured["messages"] == [{"role": "user", "content": "usr"}]


def test_complete_json_enables_adaptive_thinking_for_non_haiku():
    captured = {}

    class Adapter(AnthropicClient):
        def _create_message(self, **kwargs):
            captured.update(kwargs)
            return FakeMessage('{"ok": true}')

    Adapter(client=object()).complete_json(
        model="claude-sonnet-4-6", system="s", user="u", schema={"type": "object"}
    )
    assert captured.get("thinking") == {"type": "adaptive"}


def test_complete_json_omits_thinking_for_haiku():
    captured = {}

    class Adapter(AnthropicClient):
        def _create_message(self, **kwargs):
            captured.update(kwargs)
            return FakeMessage('{"ok": true}')

    Adapter(client=object()).complete_json(
        model="claude-haiku-4-5", system="s", user="u", schema={"type": "object"}
    )
    assert "thinking" not in captured  # Haiku does not support adaptive thinking


def test_complete_json_raises_on_unparseable_text():
    class Adapter(AnthropicClient):
        def _create_message(self, **kwargs):
            return FakeMessage("not json at all")

    with pytest.raises(ValueError, match="could not parse JSON"):
        Adapter(client=object()).complete_json(
            model="claude-sonnet-4-6", system="s", user="u", schema={"type": "object"}
        )


def test_complete_json_raises_when_model_returns_json_array():
    class Adapter(AnthropicClient):
        def _create_message(self, **kwargs):
            return FakeMessage("[1, 2, 3]")

    with pytest.raises(ValueError, match="expected a JSON object"):
        Adapter(client=object()).complete_json(
            model="claude-sonnet-4-6", system="s", user="u", schema={"type": "object"}
        )


def test_complete_json_raises_on_empty_content():
    class FakeEmptyMessage:
        content = []

    class Adapter(AnthropicClient):
        def _create_message(self, **kwargs):
            return FakeEmptyMessage()

    with pytest.raises(ValueError, match="no text content block"):
        Adapter(client=object()).complete_json(
            model="claude-sonnet-4-6", system="s", user="u", schema={"type": "object"}
        )


def test_complete_json_skips_leading_thinking_block():
    class FakeThinkingBlock:
        def __init__(self, thinking):
            self.thinking = thinking  # note: no `.text` attribute, like a real thinking block

    class FakeMixedMessage:
        def __init__(self):
            self.content = [FakeThinkingBlock("reasoning..."), FakeContentBlock('{"ok": true}')]

    class Adapter(AnthropicClient):
        def _create_message(self, **kwargs):
            return FakeMixedMessage()

    out = Adapter(client=object()).complete_json(
        model="claude-sonnet-4-6", system="s", user="u", schema={"type": "object"}
    )
    assert out == {"ok": True}
