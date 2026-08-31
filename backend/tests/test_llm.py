"""Tests for prompt construction and question condensing."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.llm import SYSTEM_PROMPT, Message, build_messages, condense_question


def test_build_messages_includes_persona_and_context():
    context = "The world is the will to power—and nothing besides!"
    history = [
        Message(role="user", content="What is the will to power?"),
        Message(role="assistant", content="The fundamental drive of all life."),
    ]
    messages = build_messages(context, history, "Tell me more.")

    assert messages[0]["role"] == "system"
    assert SYSTEM_PROMPT in messages[0]["content"]
    assert context in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "What is the will to power?"}
    assert messages[2] == {"role": "assistant", "content": "The fundamental drive of all life."}
    assert messages[-1] == {"role": "user", "content": "Tell me more."}


def test_build_messages_truncates_history_to_last_ten():
    history = [
        Message(role="user" if i % 2 == 0 else "assistant", content=f"message {i}")
        for i in range(14)
    ]
    messages = build_messages("context", history, "new question")

    # system + 10 history + new message
    assert len(messages) == 12
    assert messages[1]["content"] == "message 4"
    assert messages[-2]["content"] == "message 13"


@pytest.mark.asyncio
async def test_condense_skips_llm_when_history_empty():
    client = AsyncMock()
    result = await condense_question("What is the Übermensch?", [], client=client)
    assert result == "What is the Übermensch?"
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_condense_returns_rewritten_question():
    response = MagicMock()
    response.choices[0].message.content = "  What did Nietzsche say about eternal recurrence?  "
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    history = [Message(role="user", content="Explain eternal recurrence.")]
    result = await condense_question("Why?", history, client=client)

    assert result == "What did Nietzsche say about eternal recurrence?"


@pytest.mark.asyncio
async def test_condense_falls_back_on_error():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=Exception("API down"))

    history = [Message(role="user", content="Explain eternal recurrence.")]
    result = await condense_question("Why?", history, client=client)

    assert result == "Why?"


@pytest.mark.asyncio
async def test_condense_falls_back_on_empty_completion():
    response = MagicMock()
    response.choices[0].message.content = ""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    history = [Message(role="user", content="Explain eternal recurrence.")]
    result = await condense_question("Why?", history, client=client)

    assert result == "Why?"
