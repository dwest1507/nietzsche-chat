"""LLM generation: Nietzsche persona prompt construction + Groq streaming."""

from collections.abc import AsyncIterator
from typing import Literal

from groq import AsyncGroq
from pydantic import BaseModel, Field

from .config import GROQ_API_KEY, GROQ_MODEL

# A prior turn in the conversation. `content` is bounded well above anything
# either side can legitimately produce — a user message is capped at 1000 chars
# and a generated answer at 2048 tokens — so an oversized history is a client
# sending junk, not a long conversation.
MAX_HISTORY_CONTENT = 16_000


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=MAX_HISTORY_CONTENT)


SYSTEM_PROMPT = (
    "You are Friedrich Nietzsche, the German philosopher and cultural critic.\n"
    "Embody Nietzsche's voice, style, and philosophical positions completely.\n"
    "\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Always respond in English unless the user explicitly requests another language.\n"
    "2. Base your response EXCLUSIVELY on the provided passages from Nietzsche's works.\n"
    "3. If the passages do not fully address the question, acknowledge this honestly "
    "rather than inventing.\n"
    "4. Stay faithful to what Nietzsche actually wrote — avoid speculation beyond "
    "the documented views in the passages.\n"
    "\n"
    "Stylistic guidance:\n"
    "- Be bold, provocative, and aphoristic in Nietzsche's characteristic style\n"
    "- Use vivid metaphors and poetic language drawn from the passages\n"
    "- Challenge conventional morality and comfortable beliefs\n"
    "- Write with passion and intensity\n"
    "- Do not shy away from controversial statements Nietzsche actually made\n"
    "- Use rhetorical questions effectively\n"
)

CONDENSE_PROMPT = (
    "Given the conversation below, rewrite the user's follow-up message into a single "
    "self-contained question about Nietzsche's philosophy, suitable for searching his works. "
    "Return only the rewritten question, with no preamble."
)


def build_messages(
    context: str,
    history: list[Message],
    message: str,
) -> list[dict]:
    """Construct the Groq message list from RAG context, history, and the new user message."""
    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"PASSAGES FROM MY WORKS:\n\n{context}\n\n"
        "Respond as Nietzsche would in English, grounding your answer in the "
        "provided passages."
    )
    messages: list[dict] = [{"role": "system", "content": system_content}]
    for msg in history[-10:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": message})
    return messages


async def condense_question(
    message: str,
    history: list[Message],
    client: AsyncGroq | None = None,
) -> str:
    """Rewrite a follow-up into a standalone question for retrieval.

    Skips the LLM call entirely when there is no history; falls back to the
    raw message on any failure so retrieval degrades instead of erroring.
    """
    if not history:
        return message
    if client is None:
        client = AsyncGroq(api_key=GROQ_API_KEY)

    transcript = "\n".join(f"{msg.role}: {msg.content}" for msg in history[-6:])
    try:
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": CONDENSE_PROMPT},
                {
                    "role": "user",
                    "content": f"Conversation:\n{transcript}\n\nFollow-up: {message}",
                },
            ],
            max_tokens=256,
            temperature=0.0,
        )
        condensed = (response.choices[0].message.content or "").strip()
        return condensed or message
    except Exception:  # noqa: BLE001 — retrieval falls back to the raw message
        return message


async def generate_stream(
    messages: list[dict],
    client: AsyncGroq | None = None,
) -> AsyncIterator[str]:
    """Stream plain text tokens from Groq for the given message list."""
    if client is None:
        client = AsyncGroq(api_key=GROQ_API_KEY)

    stream = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        stream=True,
        max_tokens=2048,
        temperature=0.3,
    )
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token
