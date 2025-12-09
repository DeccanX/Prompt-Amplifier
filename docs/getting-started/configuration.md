# Configuration

Prompt Amplifier offers flexible configuration options to customize behavior.

## Configuration Object

Use `PromptForgeConfig` for full control:

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.core.config import (
    PromptForgeConfig,
    EmbedderConfig,
    GeneratorConfig,
    RetrieverConfig,
)

config = PromptForgeConfig(
    # Embedder settings
    embedder=EmbedderConfig(
        provider="sentence-transformers",
        model="all-MiniLM-L6-v2"
    ),
    
    # Generator settings
    generator=GeneratorConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=2000
    ),
    
    # Retriever settings
    retriever=RetrieverConfig(
        top_k=5,
        score_threshold=0.3
    ),
    
    # Chunking settings
    chunk_size=512,
    chunk_overlap=50,
)

forge = PromptForge(config=config)
```

## Embedder Configuration

### Available Providers

| Provider | Model Options | Type |
|----------|--------------|------|
| `tfidf` | N/A | Sparse, Free |
| `bm25` | N/A | Sparse, Free |
| `sentence-transformers` | `all-MiniLM-L6-v2`, `all-mpnet-base-v2` | Dense, Free |
| `openai` | `text-embedding-3-small`, `text-embedding-3-large` | Dense, Paid |
| `google` | `text-embedding-004` | Dense, Paid |

### Examples

=== "TF-IDF"

    ```python
    config = PromptForgeConfig(
        embedder=EmbedderConfig(provider="tfidf")
    )
    ```

=== "Sentence Transformers"

    ```python
    config = PromptForgeConfig(
        embedder=EmbedderConfig(
            provider="sentence-transformers",
            model="all-mpnet-base-v2"  # Better quality
        )
    )
    ```

=== "OpenAI"

    ```python
    config = PromptForgeConfig(
        embedder=EmbedderConfig(
            provider="openai",
            model="text-embedding-3-large",  # Highest quality
            api_key="sk-..."  # Or use OPENAI_API_KEY env var
        )
    )
    ```

## Generator Configuration

### Available Providers

| Provider | Default Model | Other Models |
|----------|--------------|--------------|
| `openai` | `gpt-4o-mini` | `gpt-4o`, `gpt-4-turbo` |
| `anthropic` | `claude-3-haiku-20240307` | `claude-3-5-sonnet-20241022`, `claude-3-opus` |
| `google` | `gemini-2.0-flash` | `gemini-1.5-pro` |

### Examples

=== "OpenAI"

    ```python
    config = PromptForgeConfig(
        generator=GeneratorConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=2000
        )
    )
    ```

=== "Anthropic"

    ```python
    config = PromptForgeConfig(
        generator=GeneratorConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.5
        )
    )
    ```

=== "Google"

    ```python
    config = PromptForgeConfig(
        generator=GeneratorConfig(
            provider="google",
            model="gemini-2.0-flash"
        )
    )
    ```

## Retriever Configuration

Control how context is retrieved:

```python
config = PromptForgeConfig(
    retriever=RetrieverConfig(
        top_k=5,              # Number of chunks to retrieve
        score_threshold=0.3,  # Minimum similarity score
        strategy="vector"     # "vector", "hybrid", or "mmr"
    )
)
```

### Retrieval Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `vector` | Pure vector similarity | General use |
| `hybrid` | BM25 + Vector combined | Keyword-heavy queries |
| `mmr` | Maximum Marginal Relevance | Diverse results |

## Chunking Configuration

Control how documents are split:

```python
config = PromptForgeConfig(
    chunk_size=512,      # Characters per chunk
    chunk_overlap=50,    # Overlap between chunks
)
```

### Recommended Settings

| Document Type | Chunk Size | Overlap |
|--------------|------------|---------|
| Technical docs | 512 | 50 |
| Long articles | 1000 | 100 |
| Code files | 256 | 30 |
| FAQs | 300 | 20 |

## Direct Component Injection

For full control, inject components directly:

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import OpenAIEmbedder
from prompt_amplifier.vectorstores import ChromaStore
from prompt_amplifier.generators import AnthropicGenerator

forge = PromptForge(
    embedder=OpenAIEmbedder(
        model="text-embedding-3-small",
        batch_size=100
    ),
    vectorstore=ChromaStore(
        collection_name="my_docs",
        persist_directory="./db"
    ),
    generator=AnthropicGenerator(
        model="claude-3-5-sonnet-20241022",
        temperature=0.7
    )
)
```

## Environment Variables

Set defaults via environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google API key |
| `COHERE_API_KEY` | Cohere API key |

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Configuration Presets

### Fast (Free, Local)

```python
config = PromptForgeConfig(
    embedder=EmbedderConfig(provider="tfidf"),
    generator=GeneratorConfig(provider="openai", model="gpt-4o-mini"),
    chunk_size=256,
)
```

### Quality (Paid, Best Results)

```python
config = PromptForgeConfig(
    embedder=EmbedderConfig(
        provider="openai",
        model="text-embedding-3-large"
    ),
    generator=GeneratorConfig(
        provider="openai",
        model="gpt-4o"
    ),
    retriever=RetrieverConfig(top_k=10),
    chunk_size=512,
)
```

### Balanced (Good Quality, Reasonable Cost)

```python
config = PromptForgeConfig(
    embedder=EmbedderConfig(
        provider="sentence-transformers",
        model="all-mpnet-base-v2"
    ),
    generator=GeneratorConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307"
    ),
    chunk_size=400,
)
```

## Next Steps

- [Core Concepts](../guide/concepts.md) - Understand the architecture
- [Embedders Guide](../guide/embedders.md) - Deep dive into embedders

