# Retrieval Strategies

Learn how to optimize context retrieval for better prompt expansion.

## Overview

Retrieval is the process of finding relevant chunks from your knowledge base based on a query. The quality of retrieved context directly impacts the quality of expanded prompts.

## Retrieval Strategies

### 1. Vector Search (Default)

Pure semantic similarity using embeddings:

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.retrievers import VectorRetriever

forge = PromptForge()
# Uses VectorRetriever by default
```

**How it works:**

1. Convert query to embedding
2. Find chunks with highest cosine similarity
3. Return top-k results

**Best for:**

- Semantic queries ("customer happiness" → finds "satisfaction")
- General-purpose retrieval
- When using dense embedders

### 2. Hybrid Search

Combines keyword (BM25) and semantic (vector) search:

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.retrievers import HybridRetriever

forge = PromptForge(
    retriever=HybridRetriever(alpha=0.5)
)
```

**Parameters:**

- `alpha=0.0`: Pure BM25 (keywords only)
- `alpha=0.5`: Balanced (default)
- `alpha=1.0`: Pure vector (semantic only)

**Best for:**

- Queries with specific terms/names
- Technical documentation
- When exact matches matter

### 3. MMR (Maximal Marginal Relevance)

Balances relevance and diversity:

```python
from prompt_amplifier.retrievers import VectorRetriever

retriever = VectorRetriever(
    use_mmr=True,
    lambda_mult=0.5  # Diversity factor
)
```

**Parameters:**

- `lambda_mult=0.0`: Maximum diversity
- `lambda_mult=1.0`: Maximum relevance (no diversity)

**Best for:**

- Avoiding redundant results
- Getting varied perspectives
- Large knowledge bases

## Configuration

### Top-K

Number of chunks to retrieve:

```python
from prompt_amplifier.core.config import PromptForgeConfig, RetrieverConfig

config = PromptForgeConfig(
    retriever=RetrieverConfig(top_k=10)
)
```

**Guidelines:**

| Knowledge Base Size | Recommended top_k |
|--------------------|-------------------|
| < 100 chunks | 3-5 |
| 100-1000 chunks | 5-10 |
| > 1000 chunks | 10-20 |

### Score Threshold

Minimum similarity score (0-1):

```python
config = PromptForgeConfig(
    retriever=RetrieverConfig(
        top_k=10,
        score_threshold=0.3
    )
)
```

Results below threshold are filtered out.

### Filters

Filter by metadata:

```python
results = forge.search(
    query="pricing",
    k=5,
    filter={"category": "products"}
)
```

## Search vs Expand

### Search Only

Returns relevant chunks without LLM expansion:

```python
results = forge.search("customer satisfaction", k=5)

for r in results:
    print(f"Score: {r.score:.3f}")
    print(f"Content: {r.chunk.content[:100]}...")
    print()
```

### Expand

Retrieves context and generates expanded prompt:

```python
result = forge.expand("customer satisfaction")
print(result.prompt)
print(f"Context chunks used: {len(result.context_chunks)}")
```

## Optimizing Retrieval

### 1. Quality Embeddings

Better embeddings = better retrieval:

```python
# ❌ Less accurate
forge = PromptForge(embedder=TFIDFEmbedder())

# ✅ More accurate
forge = PromptForge(embedder=SentenceTransformerEmbedder())

# ✅✅ Most accurate
forge = PromptForge(embedder=OpenAIEmbedder())
```

### 2. Appropriate Chunk Size

```python
# Too small - loses context
config = PromptForgeConfig(chunk_size=100)

# Too large - retrieves irrelevant info
config = PromptForgeConfig(chunk_size=2000)

# Just right
config = PromptForgeConfig(chunk_size=400)
```

### 3. Use Hybrid for Mixed Queries

```python
# Query has specific terms + semantic meaning
query = "Q4 2024 revenue growth projections"

# Hybrid finds both:
# - Exact matches for "Q4 2024"
# - Semantic matches for "revenue growth"
forge = PromptForge(retriever=HybridRetriever(alpha=0.5))
```

### 4. Adjust Score Threshold

```python
# Too low - includes irrelevant results
score_threshold=0.1

# Too high - misses relevant results
score_threshold=0.8

# Balanced
score_threshold=0.3
```

## Debugging Retrieval

### Check What's Retrieved

```python
results = forge.search("my query", k=10)

print("Retrieved chunks:")
for i, r in enumerate(results):
    print(f"{i+1}. [{r.score:.3f}] {r.chunk.content[:50]}...")
```

### Compare Strategies

```python
from prompt_amplifier.retrievers import VectorRetriever, HybridRetriever

query = "pricing plans"

# Test vector retrieval
forge_vector = PromptForge(retriever=VectorRetriever())
results_vector = forge_vector.search(query, k=5)

# Test hybrid retrieval
forge_hybrid = PromptForge(retriever=HybridRetriever())
results_hybrid = forge_hybrid.search(query, k=5)

# Compare
print("Vector results:", [r.chunk.content[:30] for r in results_vector])
print("Hybrid results:", [r.chunk.content[:30] for r in results_hybrid])
```

## Next Steps

- [Embedders Guide](embedders.md) - Improve embedding quality
- [Vector Stores Guide](vectorstores.md) - Optimize storage

