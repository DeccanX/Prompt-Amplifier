# Embedders API

## BaseEmbedder

Abstract base class for all embedders.

```python
from prompt_amplifier.embedders import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    @property
    def dimension(self) -> int:
        return 768
    
    def embed(self, texts: list[str]) -> EmbeddingResult:
        # Your implementation
        pass
```

## TFIDFEmbedder

```python
from prompt_amplifier.embedders import TFIDFEmbedder

embedder = TFIDFEmbedder(
    max_features=10000,  # Max vocabulary size
    ngram_range=(1, 2)   # Unigrams and bigrams
)
```

## SentenceTransformerEmbedder

```python
from prompt_amplifier.embedders import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(
    model="all-MiniLM-L6-v2",  # Model name
    device="cpu"               # or "cuda"
)
```

## OpenAIEmbedder

```python
from prompt_amplifier.embedders import OpenAIEmbedder

embedder = OpenAIEmbedder(
    model="text-embedding-3-small",
    api_key="sk-...",  # Optional if env var set
    batch_size=100
)
```

## GoogleEmbedder

```python
from prompt_amplifier.embedders import GoogleEmbedder

embedder = GoogleEmbedder(
    model="text-embedding-004",
    api_key="AIza..."
)
```

