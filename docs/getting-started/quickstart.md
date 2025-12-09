# Quick Start

## Basic Usage

```python
from prompt_amplifier import PromptForge

# Initialize
forge = PromptForge()

# Add your knowledge base
forge.add_texts([
    "POC Health: Healthy means all milestones on track.",
    "Success metrics: Winscore 0-100, Feature fit percentage.",
])

# Expand a short prompt
result = forge.expand("Check deal health")

print(result.prompt)
print(f"Expansion: {result.expansion_ratio:.1f}x")
```

## With Persistent Storage

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.vectorstores import ChromaStore
from prompt_amplifier.embedders import SentenceTransformerEmbedder

forge = PromptForge(
    embedder=SentenceTransformerEmbedder(),
    vectorstore=ChromaStore(
        collection_name="my_docs",
        persist_directory="./db"
    )
)

# Load documents once
forge.load_documents("./docs/")

# Use across sessions
result = forge.expand("Summarize project status")
```

## Search Only (No LLM)

```python
# Just search without expansion
results = forge.search("customer satisfaction", k=5)

for r in results:
    print(f"[{r.score:.3f}] {r.chunk.content[:100]}...")
```

## Environment Variables

Set API keys as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
```

