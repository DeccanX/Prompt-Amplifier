# Quick Start

Get started with Prompt Amplifier in 5 minutes!

## Step 1: Install

```bash
pip install prompt-amplifier
```

## Step 2: Set Up API Keys

!!! warning "API Key Required for Expansion"
    The `expand()` function requires an LLM (OpenAI, Anthropic, or Google) to generate expanded prompts.
    Set up at least one API key before using `expand()`.

=== "OpenAI (Default)"

    ```bash
    # Set in terminal
    export OPENAI_API_KEY="sk-your-key-here"
    ```
    
    ```python
    # Or set in Python
    import os
    os.environ["OPENAI_API_KEY"] = "sk-your-key-here"
    ```

=== "Google Gemini"

    ```bash
    export GOOGLE_API_KEY="your-google-api-key"
    ```
    
    ```python
    import os
    os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
    ```

=== "Anthropic"

    ```bash
    export ANTHROPIC_API_KEY="sk-ant-your-key-here"
    ```
    
    ```python
    import os
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-your-key-here"
    ```

!!! tip "Google Colab Users"
    In Google Colab, set API keys at the top of your notebook:
    ```python
    import os
    os.environ["GOOGLE_API_KEY"] = "your-key-here"  # Free tier available!
    ```

## Step 3: Basic Usage

```python
import os
# Set your API key first!
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"

from prompt_amplifier import PromptForge
from prompt_amplifier.generators import GoogleGenerator

# Create a PromptForge instance with Google (has free tier)
forge = PromptForge(generator=GoogleGenerator())

# Add your knowledge base
forge.add_texts([
    "POC Health Status: Healthy means all milestones are on track.",
    "Warning status indicates delays or risks that need attention.",
    "Critical status means blockers that require immediate escalation.",
    "Key Metrics: Winscore (0-100), Feature Fit (%), Engagement Score.",
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

## Step 4: Search WITHOUT API Key

You can search your knowledge base **without any API key**:

```python
from prompt_amplifier import PromptForge

# No API key needed for search!
forge = PromptForge()

# Add documents
forge.add_texts([
    "POC Health Status: Healthy means all milestones are on track.",
    "Warning status indicates delays or risks that need attention.", 
    "Winscore measures deal probability from 0-100.",
])

# Search returns similar chunks (no LLM needed)
results = forge.search("deal health")

for r in results.results:
    print(f"[{r.score:.3f}] {r.chunk.content[:80]}...")
```

## Step 5: Understanding the Output

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

## Step 6: Load Documents

Instead of adding text manually, load from files:

```python
# Load a single file
forge.load_documents("./data/manual.pdf")

# Load a directory (all supported formats)
forge.load_documents("./docs/")

# Supported formats: PDF, DOCX, Excel, CSV, TXT, JSON
```

## Complete Example (Google Colab Ready)

Copy this into Google Colab:

```python
# Step 1: Install
!pip install prompt-amplifier google-generativeai -q

# Step 2: Set API Key
import os
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"  # Get from https://aistudio.google.com/

# Step 3: Import
from prompt_amplifier import PromptForge
from prompt_amplifier.generators import GoogleGenerator

# Step 4: Create PromptForge with Google Gemini
forge = PromptForge(generator=GoogleGenerator())

# Step 5: Add your knowledge base
forge.add_texts([
    "Company: Acme Corp is a B2B SaaS company.",
    "Product: AI-powered analytics platform.",
    "Pricing: Starter $99/mo, Pro $299/mo, Enterprise custom.",
    "Support: 24/7 for Pro and Enterprise plans.",
    "Trial: 14-day free trial available.",
])

# Step 6: Expand prompts!
prompts = ["pricing info", "support options", "company overview"]

for prompt in prompts:
    result = forge.expand(prompt)
    print(f"\n{'='*60}")
    print(f"📝 Input: {prompt}")
    print(f"📊 Expansion: {result.expansion_ratio:.1f}x")
    print(f"{'='*60}")
    print(result.prompt[:500] + "..." if len(result.prompt) > 500 else result.prompt)
```

## Using Different LLMs

### Google Gemini (Recommended - Has Free Tier)

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-key"

from prompt_amplifier import PromptForge
from prompt_amplifier.generators import GoogleGenerator

forge = PromptForge(generator=GoogleGenerator())
```

### OpenAI GPT

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key"

from prompt_amplifier import PromptForge
from prompt_amplifier.generators import OpenAIGenerator

forge = PromptForge(generator=OpenAIGenerator(model="gpt-4o-mini"))
```

### Anthropic Claude

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-your-key"

from prompt_amplifier import PromptForge
from prompt_amplifier.generators import AnthropicGenerator

forge = PromptForge(generator=AnthropicGenerator())
```

## Using Different Embedders

### TF-IDF (Default - Free, Fast)

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import TFIDFEmbedder

forge = PromptForge(embedder=TFIDFEmbedder())
```

!!! note "TF-IDF Requires 2+ Documents"
    TF-IDF needs at least 2 documents. For single documents, use SentenceTransformerEmbedder.

### Sentence Transformers (Free, Semantic)

```python
# Install first: pip install sentence-transformers
from prompt_amplifier.embedders import SentenceTransformerEmbedder

forge = PromptForge(
    embedder=SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
)
```

### OpenAI Embeddings (Paid, High Quality)

```python
from prompt_amplifier.embedders import OpenAIEmbedder

forge = PromptForge(
    embedder=OpenAIEmbedder(model="text-embedding-3-small")
)
```

## Persistent Storage

Save embeddings for faster subsequent runs:

```python
# Install: pip install chromadb sentence-transformers
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

## Troubleshooting

### "API key for OpenAI is missing"

Set an API key before calling `expand()`:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
# OR use a different provider
from prompt_amplifier.generators import GoogleGenerator
forge = PromptForge(generator=GoogleGenerator())
```

### "TF-IDF requires at least 2 documents"

Add more documents or use a different embedder:

```python
# Option 1: Add more documents
forge.add_texts(["doc1", "doc2", "doc3"])

# Option 2: Use Sentence Transformers
from prompt_amplifier.embedders import SentenceTransformerEmbedder
forge = PromptForge(embedder=SentenceTransformerEmbedder())
forge.add_texts(["single doc works now!"])
```

## Next Steps

- [Configuration](configuration.md) - Customize behavior
- [Core Concepts](../guide/concepts.md) - Understand how it works
- [Tutorials](../tutorials/sales-intelligence.md) - Real-world examples
