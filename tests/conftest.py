"""Pytest fixtures and configuration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from prompt_amplifier.models.document import Chunk, Document

# === Sample Data Fixtures ===


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "The quarterly sales report shows a 15% increase in revenue.",
        "Customer satisfaction scores improved to 4.5 out of 5 stars.",
        "The new product launch exceeded expectations with 10,000 pre-orders.",
        "Technical support tickets decreased by 20% this quarter.",
        "Employee retention rate improved to 95% after new benefits.",
    ]


@pytest.fixture
def sample_document():
    """Single sample document."""
    return Document(
        content="This is a test document with some content for testing purposes.",
        source="test.txt",
        source_type="txt",
        metadata={"author": "test"},
    )


@pytest.fixture
def sample_documents():
    """Multiple sample documents."""
    return [
        Document(
            content="Document one about sales and revenue growth.",
            source="doc1.txt",
            source_type="txt",
        ),
        Document(
            content="Document two about customer satisfaction metrics.",
            source="doc2.txt",
            source_type="txt",
        ),
        Document(
            content="Document three about technical support and tickets.",
            source="doc3.txt",
            source_type="txt",
        ),
    ]


@pytest.fixture
def sample_chunks(sample_document):
    """Sample chunks for testing."""
    return [
        Chunk(
            content="This is chunk one.",
            document_id=sample_document.id,
            chunk_index=0,
            source=sample_document.source,
        ),
        Chunk(
            content="This is chunk two.",
            document_id=sample_document.id,
            chunk_index=1,
            source=sample_document.source,
        ),
        Chunk(
            content="This is chunk three.",
            document_id=sample_document.id,
            chunk_index=2,
            source=sample_document.source,
        ),
    ]


# === File Fixtures ===


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_txt_file(temp_dir):
    """Create a temporary text file."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("This is test content.\nLine two.\nLine three.")
    return file_path


@pytest.fixture
def temp_csv_file(temp_dir):
    """Create a temporary CSV file."""
    file_path = temp_dir / "test.csv"
    file_path.write_text("name,value,description\nitem1,100,First item\nitem2,200,Second item")
    return file_path


@pytest.fixture
def temp_json_file(temp_dir):
    """Create a temporary JSON file."""
    import json

    file_path = temp_dir / "test.json"
    data = [
        {"title": "Doc 1", "content": "Content one"},
        {"title": "Doc 2", "content": "Content two"},
    ]
    file_path.write_text(json.dumps(data))
    return file_path


# === Environment Fixtures ===


@pytest.fixture
def has_openai_key():
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def has_anthropic_key():
    """Check if Anthropic API key is available."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


# === Embedder Fixtures ===


@pytest.fixture
def tfidf_embedder():
    """TF-IDF embedder (always available)."""
    from prompt_amplifier.embedders import TFIDFEmbedder

    return TFIDFEmbedder(max_features=1000)


@pytest.fixture
def fitted_tfidf_embedder(tfidf_embedder, sample_texts):
    """Pre-fitted TF-IDF embedder."""
    tfidf_embedder.fit(sample_texts)
    return tfidf_embedder


# === Vector Store Fixtures ===


@pytest.fixture
def memory_store():
    """In-memory vector store."""
    from prompt_amplifier.vectorstores import MemoryStore

    return MemoryStore(collection_name="test")


@pytest.fixture
def populated_memory_store(memory_store, sample_chunks, fitted_tfidf_embedder):
    """Memory store with embedded chunks."""
    # Embed chunks
    fitted_tfidf_embedder.embed_chunks(sample_chunks)
    # Add to store
    memory_store.add(sample_chunks)
    return memory_store
