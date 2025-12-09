# Embedders

Embedders convert text into numerical vectors for similarity search.

## Available Embedders

| Embedder | Type | Cost | Dimension |
|----------|------|------|-----------|
| `TFIDFEmbedder` | Sparse | Free | Variable |
| `BM25Embedder` | Sparse | Free | Variable |
| `SentenceTransformerEmbedder` | Dense | Free | 384 |
| `OpenAIEmbedder` | Dense | Paid | 1536 |
| `GoogleEmbedder` | Dense | Paid | 768 |

## TF-IDF (Default)

```python
from prompt_amplifier.embedders import TFIDFEmbedder

embedder = TFIDFEmbedder()
```

Fast, keyword-based matching. Best for exact term matches.

## Sentence Transformers

```python
from prompt_amplifier.embedders import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
```

Semantic understanding. Free and runs locally.

## OpenAI

```python
from prompt_amplifier.embedders import OpenAIEmbedder

embedder = OpenAIEmbedder(
    model="text-embedding-3-small",
    api_key="sk-..."  # or set OPENAI_API_KEY
)
```

High quality. Requires API key.

## Google

```python
from prompt_amplifier.embedders import GoogleEmbedder

embedder = GoogleEmbedder(
    model="text-embedding-004",
    api_key="AIza..."  # or set GOOGLE_API_KEY
)
```

