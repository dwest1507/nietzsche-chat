"""
Pre-processing script: chunk Nietzsche's works, build FAISS + BM25 indexes.

Run once before deploying (from the backend/ directory):
    uv run python scripts/build_index.py

Reads:
  - ../../content/nietzsche/preprocessed/*.txt      19 cleaned Gutenberg works
  - ../../content/nietzsche/metadata/sources.yaml   title/translator/url per work

Writes to indexes/:
  - chunks.json     List of {"text", "work_id", "title", "translator", "url"}
  - faiss.index     FAISS inner-product index (cosine sim on normalised vecs)
  - bm25.pkl        BM25Okapi index
"""

import json
import pickle
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = Path(__file__).resolve().parent.parent
INDEXES_DIR = BACKEND_ROOT / "indexes"
PREPROCESSED_DIR = REPO_ROOT / "content" / "nietzsche" / "preprocessed"
SOURCES_YAML = REPO_ROOT / "content" / "nietzsche" / "metadata" / "sources.yaml"

# Same chunking parameters as the original vectorstore build, so retrieval
# quality carries over (~7,300 chunks of ~1,200 chars).
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


# ---------------------------------------------------------------------------
# Recursive character splitting (ported from LangChain's
# RecursiveCharacterTextSplitter with keep_separator=True)
# ---------------------------------------------------------------------------


def _split_on_separator(text: str, separator: str) -> list[str]:
    """Split text, keeping each separator attached to the start of the next piece."""
    if separator == "":
        return list(text)
    parts = re.split(f"({re.escape(separator)})", text)
    splits = [parts[i] + parts[i + 1] for i in range(1, len(parts) - 1, 2)]
    if len(parts) % 2 == 0:
        splits += parts[-1:]
    return [s for s in ([parts[0]] + splits) if s]


def _merge_splits(splits: list[str]) -> list[str]:
    """Greedily merge splits into chunks of up to CHUNK_SIZE with CHUNK_OVERLAP carryover."""
    chunks: list[str] = []
    current: list[str] = []
    total = 0
    for split in splits:
        split_len = len(split)
        if total + split_len > CHUNK_SIZE and current:
            chunk = "".join(current).strip()
            if chunk:
                chunks.append(chunk)
            # Drop from the front until within the overlap budget
            while total > CHUNK_OVERLAP or (total + split_len > CHUNK_SIZE and total > 0):
                total -= len(current[0])
                current.pop(0)
        current.append(split)
        total += split_len
    chunk = "".join(current).strip()
    if chunk:
        chunks.append(chunk)
    return chunks


def _split_text(text: str, separators: list[str] = SEPARATORS) -> list[str]:
    """Recursively split text on the first applicable separator, merging small pieces."""
    final_chunks: list[str] = []
    separator = separators[-1]
    remaining: list[str] = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining = separators[i + 1 :]
            break

    good_splits: list[str] = []
    for split in _split_on_separator(text, separator):
        if len(split) < CHUNK_SIZE:
            good_splits.append(split)
        else:
            if good_splits:
                final_chunks.extend(_merge_splits(good_splits))
                good_splits = []
            if not remaining:
                final_chunks.append(split)
            else:
                final_chunks.extend(_split_text(split, remaining))
    if good_splits:
        final_chunks.extend(_merge_splits(good_splits))
    return final_chunks


# ---------------------------------------------------------------------------
# Metadata + document loading
# ---------------------------------------------------------------------------


def _load_sources(path: Path = SOURCES_YAML) -> dict[str, dict]:
    """Map work_id → {title, translator, url} from sources.yaml."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return {entry["id"]: entry for entry in data["sources"]}


def _fallback_title(work_id: str) -> str:
    return work_id.replace("_", " ").title()


def build_chunks() -> list[dict]:
    sources = _load_sources()
    txt_files = sorted(PREPROCESSED_DIR.glob("*.txt"))
    if not txt_files:
        print(f"ERROR: No .txt files found in {PREPROCESSED_DIR}", file=sys.stderr)
        sys.exit(1)

    all_chunks: list[dict] = []
    for txt_path in txt_files:
        work_id = txt_path.stem
        meta = sources.get(work_id, {})
        title = meta.get("title", _fallback_title(work_id))
        translator = meta.get("translator", "")
        url = meta.get("url", "")
        if work_id not in sources:
            print(f"WARNING: {work_id} missing from sources.yaml", file=sys.stderr)

        text = txt_path.read_text(encoding="utf-8")
        pieces = _split_text(text)
        for piece in pieces:
            all_chunks.append(
                {
                    "text": piece,
                    "work_id": work_id,
                    "title": title,
                    "translator": translator,
                    "url": url,
                }
            )
        print(f"  {title}: {len(pieces)} chunks")
    return all_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    INDEXES_DIR.mkdir(exist_ok=True)

    print("Chunking 19 works of Nietzsche...")
    all_chunks = build_chunks()

    print(f"\nTotal chunks: {len(all_chunks)}")
    lengths = [len(c["text"]) for c in all_chunks]
    print(
        f"  Avg: {sum(lengths) / len(lengths):.0f} chars, Min: {min(lengths)}, Max: {max(lengths)}"
    )

    # Save chunks
    with open(INDEXES_DIR / "chunks.json", "w") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    print("\nSaved chunks.json")

    # Embed with sentence-transformers
    print("\nGenerating embeddings (sentence-transformers/all-mpnet-base-v2)...")
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    # Build FAISS index (inner product = cosine sim on normalised vecs)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEXES_DIR / "faiss.index"))
    print(f"Saved faiss.index ({index.ntotal} vectors, dim={dim})")

    # Build BM25 index
    tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(INDEXES_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("Saved bm25.pkl")

    print("\nDone. Ready to start the FastAPI server.")


if __name__ == "__main__":
    main()
