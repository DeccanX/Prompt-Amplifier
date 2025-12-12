# Caching

Prompt Amplifier includes a built-in caching layer to save API costs and speed up repeated operations.

---

## Overview

Caching is **enabled by default** and caches:

- **Search results**: Avoid re-embedding and searching for the same query
- **Expand results**: Avoid re-generating prompts for the same input

---

## Quick Start

```python
from prompt_amplifier import PromptForge

# Caching is enabled by default!
forge = PromptForge()
forge.add_texts(["Document 1", "Document 2", "Document 3"])

# First search - hits the embedder and vector store
result1 = forge.search("my query")  # Cache MISS

# Second search - returns cached result instantly
result2 = forge.search("my query")  # Cache HIT (much faster!)

# Check cache statistics
stats = forge.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
# Output: Hit rate: 50.0%
```

---

## Cache Types

### Memory Cache (Default)

Fast, in-memory caching. Data is lost when the program exits.

```python
from prompt_amplifier import PromptForge, MemoryCache, CacheConfig

# Customize memory cache
cache = MemoryCache(CacheConfig(
    ttl_seconds=3600,  # 1 hour TTL
    max_size=1000,     # Max 1000 entries
))

forge = PromptForge(cache=cache)
```

### Disk Cache

Persistent caching across program restarts.

```python
from prompt_amplifier import PromptForge, DiskCache, CacheConfig

# Cache persists to disk
cache = DiskCache(CacheConfig(
    cache_dir="./.prompt_cache",
    ttl_seconds=86400,  # 24 hours
    max_size=500,
))

forge = PromptForge(cache=cache)
```

---

## Configuration Options

```python
from prompt_amplifier import CacheConfig

config = CacheConfig(
    enabled=True,           # Enable/disable caching
    cache_embeddings=True,  # Cache embedding results
    cache_generations=True, # Cache LLM generation results  
    cache_searches=True,    # Cache search results
    ttl_seconds=3600,       # Time-to-live (0 = no expiry)
    max_size=1000,          # Max entries (0 = unlimited)
    cache_dir=".cache",     # Directory for DiskCache
)
```

---

## Disabling Cache

### Globally

```python
forge = PromptForge(enable_cache=False)
```

### Per-Operation

```python
# Bypass cache for this specific call
result = forge.expand("my prompt", use_cache=False)
result = forge.search("my query", use_cache=False)
```

---

## Cache Management

### View Statistics

```python
stats = forge.get_cache_stats()
print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")
print(f"Hit Rate: {stats['hit_rate']:.1%}")
print(f"Size: {stats['size']} entries")
```

**Example output:**
```
Hits: 45
Misses: 12
Hit Rate: 78.9%
Size: 12 entries
```

### Clear Cache

```python
# Clear all cached entries
cleared = forge.clear_cache()
print(f"Cleared {cleared} entries")
```

### Check Cache Status

```python
print(f"Caching enabled: {forge.cache_enabled}")
```

---

## Cache Key Generation

Cache keys are automatically generated based on:

- **Search**: query + top_k + document count
- **Expand**: prompt + top_k + system_prompt + document count

This ensures:
- Same query with different top_k → different cache entries
- Adding new documents → cache entries become stale

---

## Best Practices

### 1. Use Disk Cache for Production

```python
# Persist cache across restarts
forge = PromptForge(cache=DiskCache(CacheConfig(
    cache_dir="./cache",
    ttl_seconds=86400  # 24 hours
)))
```

### 2. Set Appropriate TTL

```python
# Short TTL for frequently changing data
CacheConfig(ttl_seconds=300)  # 5 minutes

# Long TTL for stable knowledge bases
CacheConfig(ttl_seconds=604800)  # 1 week
```

### 3. Monitor Cache Performance

```python
# Log cache stats periodically
import logging
logger = logging.getLogger("prompt_amplifier")
logger.setLevel(logging.DEBUG)

# Cache hits/misses will appear in logs
result = forge.search("query")
# DEBUG: Cache HIT for search: 'query...'
```

### 4. Clear Cache After Document Updates

```python
# Add new documents
forge.add_texts(["New document"])

# Clear cache to ensure fresh results
forge.clear_cache()
```

---

## Cost Savings Example

Without caching:
```
10 identical queries × $0.001/query = $0.01
```

With caching:
```
1 query + 9 cache hits × $0.00 = $0.001
Savings: 90%
```

For LLM expansions (more expensive):
```
Without: 10 × $0.05 = $0.50
With:    1 × $0.05 = $0.05
Savings: 90%
```

---

## Next Steps

- [Evaluation](evaluation.md) - Measure prompt quality
- [CLI Tool](cli.md) - Command-line interface

