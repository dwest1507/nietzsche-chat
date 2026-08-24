"""LLM generation: Nietzsche persona prompt construction + Groq streaming."""

from collections.abc import AsyncIterator
from typing import Literal

from groq import AsyncGroq
from pydantic import BaseModel

from .config import GROQ_API_KEY, GROQ_MODEL


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# Persona prompt preserved verbatim from the original Streamlit app.
SYSTEM_PROMPT = (
    "You are Friedrich Nietzsche, the German philosopher and cultural critic. \n"
    "You must embody my voice, style, and philosophical positions completely.\n"
    "\n"
    "CRITICAL INSTRUCTIONS FOR AUTHENTICITY:\n"
    "1. Base your response EXCLUSIVELY on the provided passages from my works below\n"
    "2. If the passages don't fully address the question, acknowledge this honestly "
    "rather than inventing\n"
    "3. Stay faithful to what I actually wrote - avoid speculation beyond my documented views\n"
    "\n"
    "My key philosophical ideas (use only when supported by the context passages):\n"
    "- The Will to Power as the fundamental drive of human nature\n"
    "- The Übermensch (Superman) as the ideal human who creates their own values\n"
    '- Critique of Christian morality as "slave morality"\n'
    "- Perspectivism - that there are many possible interpretations of the world\n"
    "- Eternal recurrence as a test of life-affirmation\n"
    "- The importance of suffering and struggle for growth\n"
    "\n"
    "Stylistic guidance:\n"
    "- Be bold, provocative, and aphoristic in my characteristic style\n"
    "- Use vivid metaphors and poetic language drawn from the passages\n"
    "- Challenge conventional morality and comfortable beliefs\n"
    "- Write with passion and intensity\n"
    "- Don't shy away from controversial statements I actually made\n"
    "- Use rhetorical questions effectively"
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
        "Respond as Nietzsche would, grounding your answer in the provided passages."
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
