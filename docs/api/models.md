# Models API

Data models used throughout Prompt Amplifier.

## Document

Represents a loaded document.

```python
from prompt_amplifier.models import Document

doc = Document(
    content="Document text content...",
    metadata={
        "source": "./file.pdf",
        "page": 1
    }
)

# Properties
doc.content      # str: Text content
doc.metadata     # dict: Metadata
doc.source       # str: Source file path
doc.char_count   # int: Character count
doc.word_count   # int: Word count
```

## Chunk

A piece of a document after chunking.

```python
from prompt_amplifier.models import Chunk

chunk = Chunk(
    id="chunk-123",
    content="Chunk text content...",
    document_id="doc-456",
    chunk_index=0,
    metadata={"source": "./file.pdf"}
)

# Properties
chunk.id           # str: Unique identifier
chunk.content      # str: Text content
chunk.document_id  # str: Parent document ID
chunk.chunk_index  # int: Position in document
chunk.metadata     # dict: Metadata
chunk.embedding    # Optional[list]: Vector embedding
chunk.has_embedding # bool: Whether embedded
chunk.embedding_dim # int: Embedding dimension
```

## ExpandResult

Result from `forge.expand()`.

```python
from prompt_amplifier.models import ExpandResult

result = forge.expand("query")

# Properties
result.prompt             # str: Expanded prompt
result.original_prompt    # str: Input prompt
result.context_chunks     # list[Chunk]: Retrieved chunks
result.expansion_ratio    # float: Length multiplier
result.retrieval_time_ms  # float: Retrieval time
result.generation_time_ms # float: LLM generation time
result.metadata           # dict: Additional metadata
```

## SearchResult

Result from `forge.search()`.

```python
from prompt_amplifier.models import SearchResult

results = forge.search("query", k=5)

for result in results:
    result.chunk    # Chunk: Matched chunk
    result.score    # float: Similarity score (0-1)
    result.metadata # dict: Additional metadata
```

## EmbeddingResult

Result from embedding operations.

```python
from prompt_amplifier.models import EmbeddingResult

# Returned by embedder.embed()
result = embedder.embed(["text1", "text2"])

result.embeddings  # list[list[float]]: Embedding vectors
result.dimension   # int: Vector dimension
result.count       # int: Number of embeddings
result.model       # str: Model used
result.is_sparse   # bool: Sparse or dense
```

## Usage Examples

### Working with Documents

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.models import Document

# Create document manually
doc = Document(
    content="This is my document content.",
    metadata={"category": "technical", "author": "John"}
)

forge = PromptForge()
forge.add_documents([doc])
```

### Working with Chunks

```python
# Access chunks after loading
for chunk in forge._chunks:
    print(f"Chunk {chunk.chunk_index}: {chunk.content[:50]}...")
    print(f"  Has embedding: {chunk.has_embedding}")
```

### Working with Results

```python
# Expand and access all result data
result = forge.expand("my query")

print(f"Original: {result.original_prompt}")
print(f"Expanded: {result.prompt[:200]}...")
print(f"Ratio: {result.expansion_ratio:.1f}x")
print(f"Retrieval: {result.retrieval_time_ms:.0f}ms")
print(f"Generation: {result.generation_time_ms:.0f}ms")

print(f"\nContext ({len(result.context_chunks)} chunks):")
for chunk in result.context_chunks:
    print(f"  - {chunk.content[:50]}...")
```

### Type Hints

All models support type hints:

```python
from prompt_amplifier.models import Document, Chunk, ExpandResult, SearchResult

def process_documents(docs: list[Document]) -> list[Chunk]:
    ...

def handle_result(result: ExpandResult) -> str:
    return result.prompt

def rank_results(results: list[SearchResult]) -> list[SearchResult]:
    return sorted(results, key=lambda r: r.score, reverse=True)
```

