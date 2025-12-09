# Vector Stores API

## BaseVectorStore

Abstract base class for all vector stores.

```python
from prompt_amplifier.vectorstores import BaseVectorStore

class MyStore(BaseVectorStore):
    def add(self, chunks, embeddings):
        pass
    
    def search(self, query_embedding, k=5):
        pass
    
    @property
    def count(self) -> int:
        pass
```

## MemoryStore

```python
from prompt_amplifier.vectorstores import MemoryStore

store = MemoryStore()
```

In-memory storage. No persistence.

## ChromaStore

```python
from prompt_amplifier.vectorstores import ChromaStore

store = ChromaStore(
    collection_name="my_collection",
    persist_directory="./chroma_db",  # Optional
    embedding_function=None            # Optional
)
```

## FAISSStore

```python
from prompt_amplifier.vectorstores import FAISSStore

store = FAISSStore(
    index_path="./faiss_index",
    dimension=384,
    index_type="Flat"  # or "IVF", "HNSW"
)
```

## Methods

### add

Add chunks with their embeddings.

```python
store.add(chunks, embeddings)
```

### search

Search for similar chunks.

```python
results = store.search(query_embedding, k=5)
```

### delete

Delete chunks by ID.

```python
store.delete(["chunk_id_1", "chunk_id_2"])
```

