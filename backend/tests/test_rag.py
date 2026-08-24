"""Tests for the RAG pipeline components and the index build script."""

import json
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Chunking + metadata (scripts/build_index.py)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_index import SOURCES_YAML, _load_sources, _split_text


def test_split_text_respects_max_size():
    para = "Man is something that is to be surpassed. " * 12  # ~500 chars
    long_text = "\n\n".join([para] * 10)
    chunks = _split_text(long_text)
    assert len(chunks) > 1
    assert all(len(c) <= 1200 for c in chunks)


def test_split_text_produces_overlap():
    """Consecutive chunks should share overlapping text."""
    sentences = [f"This is sentence number {i} about the will to power. " for i in range(60)]
    text = "".join(sentences)
    chunks = _split_text(text)
    assert len(chunks) >= 2
    # The start of chunk 2 should repeat content from the end of chunk 1
    assert chunks[1][:30] in chunks[0]


def test_split_text_short_input_is_single_chunk():
    chunks = _split_text("God is dead.")
    assert chunks == ["God is dead."]


def test_load_sources_joins_real_metadata():
    """sources.yaml must resolve every preprocessed work's metadata."""
    sources = _load_sources(SOURCES_YAML)
    assert sources["beyond_good_and_evil"]["title"] == "Beyond Good and Evil"
    assert sources["beyond_good_and_evil"]["translator"] == "Helen Zimmern"
    preprocessed = SOURCES_YAML.parent.parent / "preprocessed"
    for txt in preprocessed.glob("*.txt"):
        assert txt.stem in sources, f"{txt.stem} missing from sources.yaml"


# ---------------------------------------------------------------------------
# RAGPipeline unit tests (with mocked models/indexes)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_indexes(tmp_path):
    """Create minimal fake indexes in a temp directory."""
    import faiss as faiss_lib
    from rank_bm25 import BM25Okapi

    chunks = [
        {
            "text": "The world is the will to power. And ye yourselves are also this will "
            "to power, and nothing besides.",
            "work_id": "the_will_to_power_book_i_and_ii",
            "title": "The Will to Power, Books I and II",
            "translator": "Anthony M. Ludovici",
            "url": "https://www.gutenberg.org/",
        },
        {
            "text": "I teach you the Superman. Man is something that is to be surpassed by "
            "all who would create beyond themselves.",
            "work_id": "thus_spake_zarathustra",
            "title": "Thus Spake Zarathustra",
            "translator": "Thomas Common",
            "url": "https://www.gutenberg.org/",
        },
        {
            "text": "What is done out of love always takes place beyond good and evil. "
            "The noble soul has reverence for itself.",
            "work_id": "beyond_good_and_evil",
            "title": "Beyond Good and Evil",
            "translator": "Helen Zimmern",
            "url": "https://www.gutenberg.org/",
        },
    ]
    with open(tmp_path / "chunks.json", "w") as f:
        json.dump(chunks, f)

    dim = 768
    rng = np.random.default_rng(42)
    vectors = rng.random((len(chunks), dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss_lib.IndexFlatIP(dim)
    index.add(vectors)
    faiss_lib.write_index(index, str(tmp_path / "faiss.index"))

    corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    with open(tmp_path / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    return tmp_path, chunks


def _stub_models():
    rng = np.random.default_rng(7)
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = rng.random((1, 768)).astype(np.float32)
    mock_cross_encoder = MagicMock()
    mock_cross_encoder.predict.side_effect = lambda pairs: np.linspace(0.9, 0.1, len(pairs))
    return mock_embedder, mock_cross_encoder


def test_hybrid_search_returns_chunk_dicts(fake_indexes):
    indexes_dir, _chunks = fake_indexes
    mock_embedder, mock_cross_encoder = _stub_models()

    from app.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(
        indexes_dir=indexes_dir, embedder=mock_embedder, cross_encoder=mock_cross_encoder
    )
    results = pipeline._hybrid_search("will to power", top_k=3)

    assert len(results) > 0
    assert all("text" in r and "title" in r for r in results)


def test_reranking_orders_by_cross_encoder_score(fake_indexes):
    """The chunk with the highest cross-encoder score should appear first."""
    indexes_dir, chunks = fake_indexes
    mock_embedder, _ = _stub_models()

    mock_cross_encoder = MagicMock()
    mock_cross_encoder.predict.return_value = np.array([0.1, 0.5, 0.9])

    from app.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(
        indexes_dir=indexes_dir, embedder=mock_embedder, cross_encoder=mock_cross_encoder
    )
    reranked = pipeline._rerank("query", list(chunks), top_k=3)

    assert reranked[0] == chunks[2]


def test_retrieve_returns_top_k_dicts(fake_indexes):
    indexes_dir, _chunks = fake_indexes
    mock_embedder, mock_cross_encoder = _stub_models()

    from app.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(
        indexes_dir=indexes_dir, embedder=mock_embedder, cross_encoder=mock_cross_encoder
    )
    results = pipeline.retrieve("What is the will to power?", top_k=2)

    assert len(results) <= 2
    assert all(isinstance(r, dict) and "text" in r for r in results)


def test_sentence_filter_drops_short_fragments(fake_indexes):
    indexes_dir, _chunks = fake_indexes
    mock_embedder, mock_cross_encoder = _stub_models()

    from app.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(
        indexes_dir=indexes_dir, embedder=mock_embedder, cross_encoder=mock_cross_encoder
    )

    long_chunk = {"text": "The noble soul has reverence for itself. What is done out of love."}
    short_chunk = {"text": "CHAPTER IV."}
    filtered = pipeline._filter_short_chunks([long_chunk, short_chunk])
    assert filtered == [long_chunk]


def test_sentence_filter_falls_back_when_all_short(fake_indexes):
    indexes_dir, _chunks = fake_indexes
    mock_embedder, mock_cross_encoder = _stub_models()

    from app.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(
        indexes_dir=indexes_dir, embedder=mock_embedder, cross_encoder=mock_cross_encoder
    )

    short_chunks = [{"text": "CHAPTER IV."}, {"text": "PREFACE."}]
    assert pipeline._filter_short_chunks(short_chunks) == short_chunks
