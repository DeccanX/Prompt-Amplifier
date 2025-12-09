# Vector Stores

Vector stores persist embeddings for efficient retrieval.

## Available Stores

| Store | Type | Persistence | Best For |
|-------|------|-------------|----------|
| `MemoryStore` | Local | No | Testing |
| `ChromaStore` | Local | Yes | Development |
| `FAISSStore` | Local | Yes | Research |

## Memory Store (Default)

```python
from prompt_amplifier.vectorstores import MemoryStore

store = MemoryStore()
```

In-memory, no persistence. Good for testing.

## ChromaDB

```python
from prompt_amplifier.vectorstores import ChromaStore

store = ChromaStore(
    collection_name="my_docs",
    persist_directory="./chroma_db"
)
```

Local persistence. Popular choice for development.

## FAISS

```python
from prompt_amplifier.vectorstores import FAISSStore

store = FAISSStore(
    index_path="./faiss_index",
    dimension=384
)
```

Fast similarity search. Good for large datasets.

