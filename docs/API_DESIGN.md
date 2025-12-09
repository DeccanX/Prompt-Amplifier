# Prompt Amplifier - API Design Document

## Overview

**Prompt Amplifier** is a Python library for transforming short prompts into detailed, structured instructions using context-aware retrieval (RAG).

```bash
pip install prompt-amplifier
```

---

## Quick Start

```python
from prompt_amplifier import PromptForge

# Basic usage
forge = PromptForge()
forge.add_texts(["Your knowledge base content..."])
result = forge.expand("Short prompt")
print(result.prompt)
```

---

## Core Components

### 1. PromptForge (Main Engine)

The central orchestrator that ties all components together.

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.core.config import PromptForgeConfig, EmbedderConfig, GeneratorConfig

# Default configuration
forge = PromptForge()

# Custom configuration
config = PromptForgeConfig(
    embedder=EmbedderConfig(provider="sentence-transformers", model="all-MiniLM-L6-v2"),
    generator=GeneratorConfig(provider="openai", model="gpt-4o-mini"),
    chunk_size=512,
    chunk_overlap=50,
    top_k=5
)
forge = PromptForge(config=config)

# Or inject components directly
forge = PromptForge(
    embedder=my_embedder,
    vectorstore=my_vectorstore,
    generator=my_generator
)
```

---

## Embedders

### Available Embedders

| Embedder | Type | Cost | Dimension |
|----------|------|------|-----------|
| `TFIDFEmbedder` | Sparse | Free | Variable |
| `BM25Embedder` | Sparse | Free | Variable |
| `SentenceTransformerEmbedder` | Dense | Free | 384 |
| `OpenAIEmbedder` | Dense | Paid | 1536 |
| `GoogleEmbedder` | Dense | Paid | 768 |

### Usage Examples

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import (
    TFIDFEmbedder,
    BM25Embedder,
    SentenceTransformerEmbedder,
    OpenAIEmbedder,
    GoogleEmbedder
)

# === OPTION 1: TF-IDF (Free, keyword-based) ===
forge = PromptForge(embedder=TFIDFEmbedder())

# === OPTION 2: BM25 (Free, keyword-based, better than TF-IDF) ===
forge = PromptForge(embedder=BM25Embedder())

# === OPTION 3: Sentence Transformers (Free, semantic) ===
forge = PromptForge(
    embedder=SentenceTransformerEmbedder(
        model="all-MiniLM-L6-v2"  # or "all-mpnet-base-v2" for better quality
    )
)

# === OPTION 4: OpenAI Embeddings (Paid, high quality) ===
forge = PromptForge(
    embedder=OpenAIEmbedder(
        model="text-embedding-3-small",  # or "text-embedding-3-large"
        api_key="sk-..."  # or set OPENAI_API_KEY env var
    )
)

# === OPTION 5: Google Embeddings (Paid, high quality) ===
forge = PromptForge(
    embedder=GoogleEmbedder(
        model="text-embedding-004",
        api_key="AIza..."  # or set GOOGLE_API_KEY env var
    )
)
```

---

## Vector Stores

### Available Vector Stores

| Store | Type | Persistence | Best For |
|-------|------|-------------|----------|
| `MemoryStore` | Local | No | Testing, small datasets |
| `ChromaStore` | Local | Yes | Development, medium datasets |
| `FAISSStore` | Local | Yes | Research, benchmarking |
| `PineconeStore` | Cloud | Yes | Production (planned) |
| `QdrantStore` | Cloud/Local | Yes | Production (planned) |

### Usage Examples

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import OpenAIEmbedder
from prompt_amplifier.vectorstores import MemoryStore, ChromaStore, FAISSStore

# === OPTION 1: In-memory (default, for quick testing) ===
forge = PromptForge()
forge.load_documents("./docs/")
result = forge.expand("How's the deal going?")


# === OPTION 2: ChromaDB (local persistence) ===
forge = PromptForge(
    embedder=OpenAIEmbedder(),
    vectorstore=ChromaStore(
        collection_name="sales_docs",
        persist_directory="./chroma_db"
    )
)

# First run: loads and embeds documents
forge.load_documents("./docs/")  # Embeddings saved to ChromaDB

# Subsequent runs: instant (already embedded)
forge = PromptForge(
    embedder=OpenAIEmbedder(),
    vectorstore=ChromaStore(
        collection_name="sales_docs",
        persist_directory="./chroma_db"
    )
)
result = forge.expand("Give me POC health")  # Uses existing embeddings


# === OPTION 3: FAISS (research, benchmarking) ===
forge = PromptForge(
    embedder=SentenceTransformerEmbedder(),
    vectorstore=FAISSStore(
        index_path="./faiss_index",
        dimension=384
    )
)
```

---

## LLM Generators

### Available Generators

| Generator | Provider | Default Model |
|-----------|----------|---------------|
| `OpenAIGenerator` | OpenAI | gpt-4o-mini |
| `AnthropicGenerator` | Anthropic | claude-3-haiku-20240307 |
| `GoogleGenerator` | Google | gemini-2.0-flash |

### Usage Examples

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.generators import OpenAIGenerator, AnthropicGenerator, GoogleGenerator
from prompt_amplifier.core.config import PromptForgeConfig, GeneratorConfig

# === OPTION 1: OpenAI (default) ===
forge = PromptForge()  # Uses OpenAI by default if OPENAI_API_KEY is set

# === OPTION 2: OpenAI with custom model ===
config = PromptForgeConfig(
    generator=GeneratorConfig(provider="openai", model="gpt-4o")
)
forge = PromptForge(config=config)

# === OPTION 3: Anthropic Claude ===
config = PromptForgeConfig(
    generator=GeneratorConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
)
forge = PromptForge(config=config)

# === OPTION 4: Google Gemini ===
config = PromptForgeConfig(
    generator=GeneratorConfig(provider="google", model="gemini-2.0-flash")
)
forge = PromptForge(config=config)

# === OPTION 5: Direct injection ===
forge = PromptForge(
    generator=OpenAIGenerator(
        model="gpt-4o",
        api_key="sk-...",
        temperature=0.7,
        max_tokens=2000
    )
)
```

---

## Document Loaders

### Supported Formats

| Loader | Extensions | Description |
|--------|------------|-------------|
| `TxtLoader` | .txt | Plain text files |
| `CSVLoader` | .csv | CSV with row-per-document |
| `JSONLoader` | .json | JSON arrays or objects |
| `DocxLoader` | .docx | Microsoft Word |
| `ExcelLoader` | .xlsx | Excel spreadsheets |
| `PDFLoader` | .pdf | PDF documents |
| `DirectoryLoader` | * | Load entire directories |

### Usage Examples

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.loaders import (
    TxtLoader, CSVLoader, JSONLoader,
    DocxLoader, ExcelLoader, PDFLoader,
    DirectoryLoader
)

forge = PromptForge()

# Load individual files
forge.load_documents("./data/manual.pdf")
forge.load_documents("./data/faq.docx")
forge.load_documents("./data/products.csv")

# Load entire directory (auto-detects file types)
forge.load_documents("./docs/")

# Or add text directly
forge.add_texts([
    "Document 1 content...",
    "Document 2 content...",
])
```

---

## Retrieval Strategies

### Available Retrievers

| Retriever | Description |
|-----------|-------------|
| `VectorRetriever` | Pure vector similarity search |
| `HybridRetriever` | Combines BM25 + Vector search |

### Usage Examples

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.retrievers import VectorRetriever, HybridRetriever

# Default: Vector retrieval
forge = PromptForge()

# Hybrid retrieval (keyword + semantic)
forge = PromptForge(
    retriever=HybridRetriever(
        alpha=0.5  # Balance between BM25 (0) and Vector (1)
    )
)
```

---

## Main API Methods

### PromptForge Methods

```python
forge = PromptForge()

# === Document Management ===
forge.load_documents(path: str)           # Load from file/directory
forge.add_documents(docs: List[Document]) # Add Document objects
forge.add_texts(texts: List[str])         # Add raw text strings

# === Core Operations ===
result = forge.expand(prompt: str)        # Expand prompt with LLM
results = forge.search(query: str, k=5)   # Search without expansion

# === Properties ===
forge.chunk_count                         # Number of stored chunks
forge.embedder                            # Current embedder
forge.vectorstore                         # Current vector store
forge.generator                           # Current generator
```

### Result Objects

```python
# ExpandResult (from forge.expand())
result.prompt              # The expanded prompt (str)
result.original_prompt     # Original input prompt (str)
result.context_chunks      # Retrieved context (List[Chunk])
result.expansion_ratio     # Length multiplier (float)
result.retrieval_time_ms   # Search time in ms (float)
result.generation_time_ms  # LLM generation time in ms (float)

# SearchResult (from forge.search())
result.chunk               # The matched chunk (Chunk)
result.score               # Similarity score (float)
result.metadata            # Additional metadata (dict)
```

---

## Complete Examples

### Example 1: Sales Intelligence System

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import OpenAIEmbedder
from prompt_amplifier.vectorstores import ChromaStore
from prompt_amplifier.core.config import PromptForgeConfig, GeneratorConfig

# Configure with OpenAI for both embeddings and generation
config = PromptForgeConfig(
    generator=GeneratorConfig(provider="openai", model="gpt-4o"),
    chunk_size=512,
    top_k=5
)

forge = PromptForge(
    config=config,
    embedder=OpenAIEmbedder(),
    vectorstore=ChromaStore(
        collection_name="sales_knowledge",
        persist_directory="./db/sales"
    )
)

# Load sales documentation
forge.load_documents("./docs/sales_playbook.pdf")
forge.load_documents("./docs/product_features.docx")
forge.load_documents("./docs/objection_handling.csv")

# Expand prompts for sales reps
result = forge.expand("Customer asking about pricing")
print(result.prompt)
```

### Example 2: Research Paper Analysis

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import SentenceTransformerEmbedder
from prompt_amplifier.vectorstores import FAISSStore
from prompt_amplifier.core.config import PromptForgeConfig, GeneratorConfig

# Use local models for privacy
config = PromptForgeConfig(
    generator=GeneratorConfig(provider="anthropic"),  # Claude for generation
)

forge = PromptForge(
    config=config,
    embedder=SentenceTransformerEmbedder(model="all-mpnet-base-v2"),
    vectorstore=FAISSStore(index_path="./research_index", dimension=768)
)

# Load research papers
forge.load_documents("./papers/")

# Generate detailed analysis prompts
result = forge.expand("Compare transformer architectures")
print(f"Expansion ratio: {result.expansion_ratio:.1f}x")
print(result.prompt)
```

### Example 3: Multi-Provider Comparison

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.core.config import PromptForgeConfig, GeneratorConfig

providers = ["openai", "anthropic", "google"]

for provider in providers:
    config = PromptForgeConfig(
        generator=GeneratorConfig(provider=provider)
    )
    forge = PromptForge(config=config)
    forge.add_texts(["Your knowledge base..."])
    
    result = forge.expand("Summarize the key points")
    print(f"{provider}: {result.expansion_ratio:.1f}x in {result.generation_time_ms:.0f}ms")
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings/generation |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `GOOGLE_API_KEY` | Google API key for Gemini |
| `COHERE_API_KEY` | Cohere API key (planned) |

---

## Installation Options

```bash
# Basic installation (TF-IDF only)
pip install prompt-amplifier

# With local embeddings (Sentence Transformers)
pip install prompt-amplifier[embeddings-local]

# With OpenAI
pip install prompt-amplifier[embeddings-openai,generators-openai]

# With all features
pip install prompt-amplifier[all]

# Development
pip install prompt-amplifier[dev]
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        PromptForge                               │
│                     (Main Orchestrator)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Loaders    │    │   Chunkers   │    │  Embedders   │       │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤       │
│  │ • TXT        │    │ • Recursive  │    │ • TF-IDF     │       │
│  │ • CSV        │    │ • Fixed      │    │ • BM25       │       │
│  │ • JSON       │    │ • Sentence   │    │ • SentenceTF │       │
│  │ • DOCX       │    │              │    │ • OpenAI     │       │
│  │ • Excel      │    │              │    │ • Google     │       │
│  │ • PDF        │    │              │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│          │                  │                   │                │
│          ▼                  ▼                   ▼                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Vector Store                          │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │    │
│  │  │ Memory  │  │ Chroma  │  │  FAISS  │  │Pinecone │    │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     Retrievers                           │    │
│  │         ┌─────────────┐    ┌─────────────┐              │    │
│  │         │   Vector    │    │   Hybrid    │              │    │
│  │         └─────────────┘    └─────────────┘              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     Generators                           │    │
│  │     ┌────────┐   ┌───────────┐   ┌────────────┐        │    │
│  │     │ OpenAI │   │ Anthropic │   │   Google   │        │    │
│  │     └────────┘   └───────────┘   └────────────┘        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│                    ┌──────────────────┐                         │
│                    │  Expanded Prompt │                         │
│                    └──────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024-12 | Initial release |

---

## License

Apache 2.0 License

---

## Links

- **PyPI**: https://pypi.org/project/prompt-amplifier/
- **GitHub**: https://github.com/DeccanX/prompt-amplifier
- **Documentation**: https://prompt-amplifier.readthedocs.io

