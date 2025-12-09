# PromptForge API

The main class for prompt amplification.

## PromptForge

```python
from prompt_amplifier import PromptForge

forge = PromptForge(
    config=None,      # PromptForgeConfig
    embedder=None,    # Custom embedder
    vectorstore=None, # Custom vector store
    generator=None    # Custom generator
)
```

## Methods

### load_documents

Load documents from a file or directory.

```python
forge.load_documents("./docs/")
forge.load_documents("./data/manual.pdf")
```

### add_texts

Add raw text strings.

```python
forge.add_texts([
    "Document 1 content...",
    "Document 2 content...",
])
```

### expand

Expand a short prompt using RAG.

```python
result = forge.expand("Short prompt")

print(result.prompt)           # Expanded prompt
print(result.expansion_ratio)  # e.g., 50.0x
print(result.generation_time_ms)
```

### search

Search without expansion.

```python
results = forge.search("query", k=5)

for r in results:
    print(r.score, r.chunk.content)
```

## Properties

- `forge.chunk_count` - Number of stored chunks
- `forge.embedder` - Current embedder
- `forge.vectorstore` - Current vector store
- `forge.generator` - Current generator

