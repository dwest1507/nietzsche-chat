"""POST /api/chat — question condensing + RAG retrieval + streamed generation.

Stream protocol (Vercel AI SDK data stream v1 line format):
    2:[{"title", "translator", "url", "text"}, ...]   source passages, sent first
    0:"token"                                          one line per generated token
    d:{"finishReason": "stop"}                         end of stream
    3:"Generation failed"                              error (replaces d: line)
"""

import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..llm import Message, build_messages, condense_question, generate_stream
from ..rag.pipeline import get_pipeline
from ..ratelimit import CHAT_RATE_LIMIT, limiter
from ..security import require_shared_secret

logger = logging.getLogger("uvicorn.error")

# Generous headroom over the 10 turns the frontend sends and the API uses.
MAX_HISTORY_MESSAGES = 50

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    # Only the last 10 turns are ever used; the cap keeps an unbounded history
    # from being parsed into memory just to be thrown away.
    history: list[Message] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)


# The dependency runs before the decorator's limit check, which is what makes the
# forwarded address in the limiter key safe to trust. See app/ratelimit.py.
@router.post("/chat", dependencies=[Depends(require_shared_secret)])
@limiter.limit(CHAT_RATE_LIMIT)
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    async def generate():
        try:
            query = await condense_question(body.message, body.history)
            # Retrieval is synchronous and CPU-bound (embedding + cross-encoder
            # inference, and the model load itself on the first call). Running it
            # inline would stall the event loop for its whole duration, freezing
            # every other request — including /health — so hand it to a worker.
            chunks = await run_in_threadpool(lambda: get_pipeline().retrieve(query))

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
        # Never leak provider errors into the stream; the client sees a
        # generic failure while the traceback goes to the server log.
        except Exception:
            logger.exception("Chat generation failed")
            yield '3:"Generation failed"\n'

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
