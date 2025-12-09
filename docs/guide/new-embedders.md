# Additional Embedders

Prompt Amplifier v0.2.0 adds support for several new embedding providers beyond the core TF-IDF, BM25, Sentence Transformers, OpenAI, and Google embedders.

## Cohere Embeddings

High-quality embeddings from Cohere with built-in reranking support.

```python
from prompt_amplifier.embedders import CohereEmbedder

# Basic usage
embedder = CohereEmbedder(
    api_key="your-cohere-api-key",  # or COHERE_API_KEY env var
    model="embed-english-v3.0",
)

result = embedder.embed(["Your text here"])
print(f"Dimension: {result.dimension}")  # 1024
```

### Cohere Reranker

Improve retrieval quality by reranking results:

```python
from prompt_amplifier.embedders import CohereRerankEmbedder

reranker = CohereRerankEmbedder(
    api_key="your-cohere-api-key",
    model="rerank-english-v2.0",
)

# Rerank retrieved documents
query = "What is machine learning?"
documents = [
    "ML is a subset of AI",
    "Weather forecast for tomorrow",
    "Deep learning uses neural networks",
]

reranked = reranker.rerank(query, documents, top_n=2)
for doc, score in reranked:
    print(f"{score:.3f}: {doc}")
```

### Installation

```bash
pip install prompt-amplifier[embeddings-cohere]
```

---

## Voyage AI Embeddings

Specialized embeddings optimized for retrieval and RAG applications.

```python
from prompt_amplifier.embedders import VoyageEmbedder

embedder = VoyageEmbedder(
    api_key="your-voyage-api-key",  # or VOYAGE_API_KEY env var
    model="voyage-2",  # or voyage-lite-02-instruct
)

result = embedder.embed([
    "Document for indexing",
    "Another document",
])
```

### Available Models

| Model | Dimension | Best For |
|-------|-----------|----------|
| `voyage-2` | 1024 | General purpose |
| `voyage-large-2` | 1536 | Maximum quality |
| `voyage-code-2` | 1536 | Code retrieval |
| `voyage-lite-02-instruct` | 1024 | Fast inference |

### Installation

```bash
pip install prompt-amplifier[embeddings-voyage]
```

---

## Jina AI Embeddings

Multilingual embeddings with excellent performance across 100+ languages.

```python
from prompt_amplifier.embedders import JinaEmbedder

embedder = JinaEmbedder(
    api_key="your-jina-api-key",  # or JINA_API_KEY env var
    model="jina-embeddings-v2-base-en",
)

# Multilingual example
result = embedder.embed([
    "Hello world",
    "Hallo Welt",
    "Bonjour le monde",
])
```

### Available Models

| Model | Languages | Dimension |
|-------|-----------|-----------|
| `jina-embeddings-v2-base-en` | English | 768 |
| `jina-embeddings-v2-base-de` | German | 768 |
| `jina-embeddings-v2-base-multilingual` | 100+ | 768 |
| `jina-embeddings-v2-small-en` | English | 512 |

### Installation

```bash
pip install prompt-amplifier[embeddings-jina]
```

---

## Mistral AI Embeddings

European AI embeddings with strong multilingual capabilities.

```python
from prompt_amplifier.embedders import MistralEmbedder

embedder = MistralEmbedder(
    api_key="your-mistral-api-key",  # or MISTRAL_API_KEY env var
    model="mistral-embed",
)

result = embedder.embed(["Your text here"])
print(f"Dimension: {result.dimension}")  # 1024
```

### Features

- GDPR compliant (EU-based)
- Strong multilingual support
- Competitive pricing
- Fast inference

### Installation

```bash
pip install prompt-amplifier[embeddings-mistral]
```

---

## FastEmbed (Local)

Fast, local embeddings using ONNX runtime - no API keys needed!

```python
from prompt_amplifier.embedders import FastEmbedEmbedder

embedder = FastEmbedEmbedder(
    model_name="BAAI/bge-small-en-v1.5",
)

# Runs completely locally
result = embedder.embed([
    "This runs on your machine",
    "No API calls needed",
])
```

### Available Models

| Model | Dimension | Speed |
|-------|-----------|-------|
| `BAAI/bge-small-en-v1.5` | 384 | Fastest |
| `BAAI/bge-base-en-v1.5` | 768 | Balanced |
| `BAAI/bge-large-en-v1.5` | 1024 | Best quality |

### Installation

```bash
pip install fastembed
```

---

## Comparison Table

| Provider | Local | Cost | Quality | Speed | Multilingual |
|----------|-------|------|---------|-------|--------------|
| TF-IDF | ✅ | Free | ⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
| BM25 | ✅ | Free | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
| Sentence Transformers | ✅ | Free | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ |
| FastEmbed | ✅ | Free | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| OpenAI | ❌ | $$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| Google | ❌ | $$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| Cohere | ❌ | $$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| Voyage | ❌ | $$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| Jina | ❌ | $$ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅✅ |
| Mistral | ❌ | $$ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅✅ |

---

## Choosing an Embedder

### For Research/Prototyping
```python
# Free, local, good quality
from prompt_amplifier.embedders import SentenceTransformerEmbedder
embedder = SentenceTransformerEmbedder()
```

### For Production (Budget)
```python
# Fast, local, ONNX optimized
from prompt_amplifier.embedders import FastEmbedEmbedder
embedder = FastEmbedEmbedder()
```

### For Production (Quality)
```python
# Best quality, paid API
from prompt_amplifier.embedders import CohereEmbedder
embedder = CohereEmbedder()
```

### For Multilingual
```python
# Best multilingual support
from prompt_amplifier.embedders import JinaEmbedder
embedder = JinaEmbedder(model="jina-embeddings-v2-base-multilingual")
```

### For GDPR Compliance
```python
# EU-based provider
from prompt_amplifier.embedders import MistralEmbedder
embedder = MistralEmbedder()
```

### For Code Search
```python
# Optimized for code
from prompt_amplifier.embedders import VoyageEmbedder
embedder = VoyageEmbedder(model="voyage-code-2")
```

