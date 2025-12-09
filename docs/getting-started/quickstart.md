# Quick Start

Get started with Prompt Amplifier in 5 minutes!

## Step 1: Install

```bash
pip install prompt-amplifier
```

## Step 2: Basic Usage

```python
from prompt_amplifier import PromptForge

# Create a PromptForge instance
forge = PromptForge()

# Add your knowledge base
forge.add_texts([
    "POC Health Status: Healthy means all milestones are on track with positive customer engagement.",
    "Warning status indicates delays or risks that need attention.",
    "Critical status means blockers that require immediate escalation.",
    "Key Metrics: Winscore (0-100), Feature Fit (%), Customer Engagement Score.",
])

# Expand a short prompt
result = forge.expand("Check the deal health")

print("=" * 60)
print("EXPANDED PROMPT:")
print("=" * 60)
print(result.prompt)
print()
print(f"📊 Expansion ratio: {result.expansion_ratio:.1f}x")
print(f"⏱️ Generation time: {result.generation_time_ms:.0f}ms")
```

## Step 3: Understanding the Output

The `expand()` method returns an `ExpandResult` object:

```python
result = forge.expand("Your prompt")

# The expanded prompt (string)
print(result.prompt)

# Original prompt
print(result.original_prompt)  # "Your prompt"

# Expansion statistics
print(result.expansion_ratio)     # e.g., 45.2 (45x longer)
print(result.retrieval_time_ms)   # e.g., 12.5 (milliseconds)
print(result.generation_time_ms)  # e.g., 2500 (milliseconds)

# Retrieved context
for chunk in result.context_chunks:
    print(f"- {chunk.content[:50]}...")
```

## Step 4: Search Without Expansion

You can also just search your knowledge base without LLM expansion:

```python
# Search returns similar chunks
results = forge.search("customer satisfaction", k=3)

for r in results:
    print(f"[{r.score:.3f}] {r.chunk.content[:80]}...")
```

## Step 5: Load Documents

Instead of adding text manually, load from files:

```python
# Load a single file
forge.load_documents("./data/manual.pdf")

# Load a directory (all supported formats)
forge.load_documents("./docs/")

# Supported formats: PDF, DOCX, Excel, CSV, TXT, JSON
```

## Complete Example

```python
from prompt_amplifier import PromptForge

# Initialize
forge = PromptForge()

# Build knowledge base
forge.add_texts([
    "Company: Acme Corp is a B2B SaaS company.",
    "Product: AI-powered analytics platform.",
    "Pricing: Starter $99/mo, Pro $299/mo, Enterprise custom.",
    "Support: 24/7 for Pro and Enterprise plans.",
    "Trial: 14-day free trial available.",
])

# Test different prompts
prompts = [
    "pricing info",
    "how to contact support",
    "company overview",
]

for prompt in prompts:
    result = forge.expand(prompt)
    print(f"\n{'='*60}")
    print(f"Input: {prompt}")
    print(f"Expansion: {result.expansion_ratio:.1f}x")
    print(f"{'='*60}")
    print(result.prompt[:500] + "...")
```

## Using Different Embedders

### TF-IDF (Default - Free, Fast)

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import TFIDFEmbedder

forge = PromptForge(embedder=TFIDFEmbedder())
```

### Sentence Transformers (Free, Semantic)

```python
from prompt_amplifier.embedders import SentenceTransformerEmbedder

forge = PromptForge(
    embedder=SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
)
```

### OpenAI (Paid, High Quality)

```python
from prompt_amplifier.embedders import OpenAIEmbedder

forge = PromptForge(
    embedder=OpenAIEmbedder(model="text-embedding-3-small")
)
```

## Using Different LLMs

### OpenAI GPT (Default)

```python
from prompt_amplifier.core.config import PromptForgeConfig, GeneratorConfig

config = PromptForgeConfig(
    generator=GeneratorConfig(provider="openai", model="gpt-4o")
)
forge = PromptForge(config=config)
```

### Anthropic Claude

```python
config = PromptForgeConfig(
    generator=GeneratorConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
)
forge = PromptForge(config=config)
```

### Google Gemini

```python
config = PromptForgeConfig(
    generator=GeneratorConfig(provider="google", model="gemini-2.0-flash")
)
forge = PromptForge(config=config)
```

## Persistent Storage

Save embeddings for faster subsequent runs:

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.vectorstores import ChromaStore
from prompt_amplifier.embedders import SentenceTransformerEmbedder

# First run: creates and saves embeddings
forge = PromptForge(
    embedder=SentenceTransformerEmbedder(),
    vectorstore=ChromaStore(
        collection_name="my_docs",
        persist_directory="./chroma_db"
    )
)
forge.load_documents("./docs/")  # Embeddings saved to disk

# Later runs: loads existing embeddings instantly
forge = PromptForge(
    embedder=SentenceTransformerEmbedder(),
    vectorstore=ChromaStore(
        collection_name="my_docs",
        persist_directory="./chroma_db"
    )
)
# No need to reload - embeddings already there!
result = forge.expand("My query")
```

## Next Steps

- [Configuration](configuration.md) - Customize behavior
- [Core Concepts](../guide/concepts.md) - Understand how it works
- [Tutorials](../tutorials/sales-intelligence.md) - Real-world examples
