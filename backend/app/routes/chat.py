"""POST /api/chat — question condensing + RAG retrieval + streamed generation.

Stream protocol (Vercel AI SDK data stream v1 line format):
    2:[{"title", "translator", "url", "text"}, ...]   source passages, sent first
    0:"token"                                          one line per generated token
    d:{"finishReason": "stop"}                         end of stream
    3:"Generation failed"                              error (replaces d: line)
"""

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..llm import Message, build_messages, condense_question, generate_stream
from ..rag.pipeline import get_pipeline

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    history: list[Message] = Field(default_factory=list)


@router.post("/chat")
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    async def generate():
        try:
            query = await condense_question(body.message, body.history)
            chunks = get_pipeline().retrieve(query)

            sources = [
                {
                    "title": c.get("title", ""),
                    "translator": c.get("translator", ""),
                    "url": c.get("url", ""),
                    "text": c["text"],
                }
                for c in chunks
            ]
            yield f"2:{json.dumps(sources)}\n"

            context = "\n\n---\n\n".join(c["text"] for c in chunks)
            messages = build_messages(context, body.history, body.message)
            async for token in generate_stream(messages):
                yield f"0:{json.dumps(token)}\n"
            yield f"d:{json.dumps({'finishReason': 'stop'})}\n"
        except Exception:  # noqa: BLE001 — never leak provider errors into the stream
            yield '3:"Generation failed"\n'

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
