# Installation

This guide covers all installation options for Prompt Amplifier.

---

## Basic Installation

```bash
pip install prompt-amplifier
```

**What you get:**

- ✅ Core `PromptForge` class
- ✅ TF-IDF embedder (free, no API key)
- ✅ In-memory vector store
- ✅ Document loaders (TXT, CSV, JSON)

**What you still need:**

- ❌ API key for `expand()` function (OpenAI, Google, or Anthropic)
- ❌ PDF/DOCX support (install `loaders` extra)
- ❌ Semantic embeddings (install `embeddings-local` extra)

---

## Installation Options

### Full Installation (Recommended)

```bash
pip install prompt-amplifier[all]
```

**What you get:** Everything! All loaders, embedders, vector stores, and generators.

**Size:** ~2GB (includes ML models)

---

### Minimal + Google Gemini (Lightweight)

```bash
pip install prompt-amplifier google-generativeai
```

**What you get:**

- Core library
- Google Gemini generator (has free tier!)

**Size:** ~50MB

**Best for:** Quick testing, Google Colab

---

### Production Setup

```bash
pip install prompt-amplifier[loaders,embeddings-local,vectorstore-chroma]
```

**What you get:**

- All document loaders (PDF, DOCX, Excel, etc.)
- Sentence Transformers (semantic embeddings)
- ChromaDB (persistent storage)

**Size:** ~1GB

---

## Feature-Specific Extras

### Document Loaders

```bash
pip install prompt-amplifier[loaders]
```

**Adds support for:**

| Format | Library Used |
|--------|--------------|
| PDF | `pypdf` |
| DOCX | `python-docx` |
| Excel | `openpyxl` |
| Web pages | `beautifulsoup4`, `requests` |
| YouTube | `youtube-transcript-api` |
| RSS feeds | `feedparser` |

---

### Embedders

=== "Local (Free)"

    ```bash
    pip install prompt-amplifier[embeddings-local]
    ```
    
    **Adds:**
    
    - Sentence Transformers (semantic search)
    - BM25 (keyword search)
    
    **Models used:**
    
    - `all-MiniLM-L6-v2` (384 dimensions, fast)
    - `all-mpnet-base-v2` (768 dimensions, better quality)

=== "OpenAI"

    ```bash
    pip install prompt-amplifier[embeddings-openai]
    ```
    
    **Adds:** OpenAI embeddings (`text-embedding-3-small`, `text-embedding-3-large`)
    
    **Requires:** `OPENAI_API_KEY`

=== "Google"

    ```bash
    pip install prompt-amplifier[embeddings-google]
    ```
    
    **Adds:** Google embeddings (`text-embedding-004`)
    
    **Requires:** `GOOGLE_API_KEY`

=== "Cohere"

    ```bash
    pip install prompt-amplifier[embeddings-cohere]
    ```
    
    **Adds:** Cohere embeddings (`embed-english-v3.0`)
    
    **Requires:** `COHERE_API_KEY`

---

### Vector Stores

=== "ChromaDB"

    ```bash
    pip install prompt-amplifier[vectorstore-chroma]
    ```
    
    **Use case:** Local persistent storage
    
    ```python
    from prompt_amplifier.vectorstores import ChromaStore
    
    store = ChromaStore(
        collection_name="my_docs",
        persist_directory="./chroma_db"
    )
    ```

=== "FAISS"

    ```bash
    pip install prompt-amplifier[vectorstore-faiss]
    ```
    
    **Use case:** High-performance similarity search
    
    ```python
    from prompt_amplifier.vectorstores import FAISSStore
    
    store = FAISSStore(index_path="./faiss_index")
    ```

---

### LLM Generators

=== "OpenAI"

    ```bash
    pip install prompt-amplifier[generators-openai]
    ```
    
    **Requires:** `OPENAI_API_KEY`
    
    ```python
    from prompt_amplifier.generators import OpenAIGenerator
    forge = PromptForge(generator=OpenAIGenerator(model="gpt-4o-mini"))
    ```

=== "Anthropic"

    ```bash
    pip install prompt-amplifier[generators-anthropic]
    ```
    
    **Requires:** `ANTHROPIC_API_KEY`
    
    ```python
    from prompt_amplifier.generators import AnthropicGenerator
    forge = PromptForge(generator=AnthropicGenerator())
    ```

=== "Google"

    ```bash
    pip install prompt-amplifier[generators-google]
    # or just: pip install google-generativeai
    ```
    
    **Requires:** `GOOGLE_API_KEY` (free tier available!)
    
    ```python
    from prompt_amplifier.generators import GoogleGenerator
    forge = PromptForge(generator=GoogleGenerator())
    ```

=== "Ollama (Local)"

    ```bash
    pip install prompt-amplifier[generators-ollama]
    ```
    
    **Requires:** [Ollama](https://ollama.ai/) running locally
    
    ```python
    from prompt_amplifier.generators import OllamaGenerator
    forge = PromptForge(generator=OllamaGenerator(model="llama3.2"))
    ```

---

## Verify Installation

Run this to verify everything works:

```python
# Test basic import
from prompt_amplifier import PromptForge
print("✅ Core library installed")

# Test with sample data
forge = PromptForge()
forge.add_texts(["Test document 1", "Test document 2"])
print(f"✅ Added {forge.chunk_count} chunks")

# Test search (no API key needed)
results = forge.search("test")
print(f"✅ Search works! Found {len(results.results)} results")

# Check available extras
try:
    from prompt_amplifier.embedders import SentenceTransformerEmbedder
    print("✅ Sentence Transformers available")
except ImportError:
    print("❌ Install with: pip install prompt-amplifier[embeddings-local]")

try:
    from prompt_amplifier.vectorstores import ChromaStore
    print("✅ ChromaDB available")
except ImportError:
    print("❌ Install with: pip install prompt-amplifier[vectorstore-chroma]")

try:
    from prompt_amplifier.generators import GoogleGenerator
    print("✅ Google generator available")
except ImportError:
    print("❌ Install with: pip install google-generativeai")
```

**Expected Output:**
```
✅ Core library installed
✅ Added 2 chunks
✅ Search works! Found 2 results
✅ Sentence Transformers available (or install message)
✅ ChromaDB available (or install message)
✅ Google generator available (or install message)
```

---

## Google Colab Quick Setup

For Google Colab, run these cells:

**Cell 1: Install**
```python
!pip install prompt-amplifier google-generativeai -q
print("✅ Installed!")
```

**Cell 2: Set API Key**
```python
import os
os.environ["GOOGLE_API_KEY"] = "your-key-from-aistudio.google.com"
print("✅ API key set!")
```

**Cell 3: Test**
```python
from prompt_amplifier import PromptForge
from prompt_amplifier.generators import GoogleGenerator

forge = PromptForge(generator=GoogleGenerator())
forge.add_texts(["Hello world", "Testing prompt amplifier"])

result = forge.expand("test the system")
print(result.prompt)
```

---

## Troubleshooting

### "No module named 'prompt_amplifier'"

```bash
# Make sure you installed it
pip install prompt-amplifier

# If using virtual environment, activate it first
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

### "ImportError: openai is required"

```bash
pip install openai
# or install the extra
pip install prompt-amplifier[generators-openai]
```

### "ImportError: sentence_transformers is required"

```bash
pip install sentence-transformers
# or install the extra
pip install prompt-amplifier[embeddings-local]
```

### "No space left on device" (Colab)

Sentence Transformers downloads ~400MB models. In Colab:

```python
# Use smaller model
from prompt_amplifier.embedders import SentenceTransformerEmbedder
embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")  # Smallest
```

---

## Next Steps

- [Quick Start](quickstart.md) - Get running in 5 minutes
- [Configuration](configuration.md) - Customize behavior
- [Core Concepts](../guide/concepts.md) - Understand how it works
