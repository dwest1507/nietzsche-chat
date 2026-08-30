import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# Shared with the Next.js route handler, the backend's only legitimate caller.
# Required: a missing value must fail loudly at import rather than leave the
# public Railway URL open to anyone who finds it.
BACKEND_SHARED_SECRET: str = os.environ["BACKEND_SHARED_SECRET"]
ALLOWED_ORIGINS: list[str] = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

INDEXES_DIR = Path(__file__).parent.parent / "indexes"
