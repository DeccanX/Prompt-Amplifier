# Prompt Amplifier

<p align="center">
  <strong>Transform short prompts into detailed, structured instructions using RAG</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/prompt-amplifier/">
    <img src="https://img.shields.io/pypi/v/prompt-amplifier.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/prompt-amplifier/">
    <img src="https://img.shields.io/pypi/pyversions/prompt-amplifier.svg" alt="Python versions">
  </a>
  <a href="https://github.com/DeccanX/Prompt-Amplifier/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
</p>

---

## What is Prompt Amplifier?

**Prompt Amplifier** is a Python library that transforms brief user inputs into comprehensive, well-structured prompts using Retrieval-Augmented Generation (RAG).

Instead of manually crafting detailed prompts, you can:

1. **Load your domain knowledge** (documents, web pages, videos)
2. **Write a simple prompt** ("How's the deal going?")
3. **Get a detailed prompt** with relevant context, structure, and instructions

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install the Library

```bash
pip install prompt-amplifier
```

**What this does:** Installs the core Prompt Amplifier library with default dependencies (TF-IDF embedder, in-memory vector store).

---

### Step 2: Set Up Your API Key

!!! warning "API Key Required"
    The `expand()` function requires an LLM to generate expanded prompts. You need at least one API key.

=== "Google Gemini (Recommended - Free Tier)"

    ```python
    import os
    os.environ["GOOGLE_API_KEY"] = "your-key-here"
    ```
    
    **Get your free key:** [https://aistudio.google.com/](https://aistudio.google.com/)

=== "OpenAI"

    ```python
    import os
    os.environ["OPENAI_API_KEY"] = "sk-your-key-here"
    ```
    
    **Get your key:** [https://platform.openai.com/](https://platform.openai.com/)

=== "Anthropic"

    ```python
    import os
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-your-key-here"
    ```
    
    **Get your key:** [https://console.anthropic.com/](https://console.anthropic.com/)

---

### Step 3: Create Your First Expanded Prompt

```python
# Step 3a: Set your API key
import os
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"

# Step 3b: Import the library
from prompt_amplifier import PromptForge
from prompt_amplifier.generators import GoogleGenerator

# Step 3c: Create a PromptForge instance with Google Gemini
forge = PromptForge(generator=GoogleGenerator())

# Step 3d: Add your knowledge base (domain documents)
forge.add_texts([
    "POC Health Status: Healthy means all milestones are on track with positive engagement.",
    "Warning status indicates delays or risks that need immediate attention.",
    "Critical status means major blockers that require executive escalation.",
    "Key Metrics: Winscore (0-100), Feature Fit (%), Customer Engagement Score.",
])

# Step 3e: Expand a short prompt
result = forge.expand("Check the deal health")

# Step 3f: Print the result
print(result.prompt)
```

**What each line does:**

| Line | Explanation |
|------|-------------|
| `os.environ["GOOGLE_API_KEY"]` | Sets your API key for Google Gemini |
| `from prompt_amplifier import PromptForge` | Imports the main class |
| `from prompt_amplifier.generators import GoogleGenerator` | Imports Google's LLM generator |
| `PromptForge(generator=GoogleGenerator())` | Creates instance with Google Gemini |
| `forge.add_texts([...])` | Adds your domain knowledge |
| `forge.expand("...")` | Transforms short prompt into detailed one |

---

### Step 4: See the Output

**Input (what you write):**
```
Check the deal health
```

**Output (what you get):**
```
**GOAL:** Generate a comprehensive deal health assessment report.

**CONTEXT:**
Based on the POC tracking data, analyze the following aspects:
- Current health status (Healthy/Warning/Critical)
- Milestone completion percentage
- Customer engagement levels
- Risk indicators

**REQUIRED SECTIONS:**

1. **Executive Summary**
   - Overall health rating
   - Key highlights

2. **Metrics Dashboard**
   | Metric | Current Value | Status |
   |--------|--------------|--------|
   | Winscore | [0-100] | [Status] |
   | Feature Fit | [%] | [Status] |
   | Engagement | [Score] | [Status] |

3. **Risk Assessment**
   - Blockers identified
   - Mitigation strategies

4. **Recommended Actions**
   - Prioritized next steps
   - Owner assignments

**INSTRUCTIONS:**
- Use specific data points from the POC tracking system
- Highlight any warning or critical indicators
- Provide actionable recommendations
```

**Expansion Ratio:** Your 4-word input became a ~150-word structured prompt (37x expansion)!

---

## 🔍 Search Without API Key

You can search your knowledge base **without any API key**:

```python
from prompt_amplifier import PromptForge

# No API key needed!
forge = PromptForge()

# Add your documents
forge.add_texts([
    "POC Health: Healthy means milestones on track.",
    "Winscore measures deal probability from 0-100.",
    "Feature Fit shows product-customer alignment.",
])

# Search for relevant chunks
results = forge.search("deal probability")

# Print results
for r in results.results:
    print(f"Score: {r.score:.3f}")
    print(f"Content: {r.chunk.content}")
    print("---")
```

**Expected Output:**
```
Score: 0.892
Content: Winscore measures deal probability from 0-100.
---
Score: 0.654
Content: POC Health: Healthy means milestones on track.
---
Score: 0.423
Content: Feature Fit shows product-customer alignment.
---
```

---

## 📁 Load Documents from Files

Instead of adding text manually, load from files:

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.generators import GoogleGenerator

forge = PromptForge(generator=GoogleGenerator())

# Load a single file
forge.load_documents("./data/manual.pdf")

# Load an entire directory
forge.load_documents("./docs/")

# Check how many documents were loaded
print(f"Loaded {forge.document_count} documents")
print(f"Created {forge.chunk_count} searchable chunks")
```

**Supported Formats:**

| Format | Extension | Example |
|--------|-----------|---------|
| Plain Text | `.txt` | notes.txt |
| PDF | `.pdf` | report.pdf |
| Word | `.docx` | document.docx |
| Excel | `.xlsx` | data.xlsx |
| CSV | `.csv` | records.csv |
| JSON | `.json` | config.json |

---

## 🎯 Complete Google Colab Example

Copy this entire code block into Google Colab:

```python
# ============================================
# PROMPT AMPLIFIER - COMPLETE EXAMPLE
# ============================================

# Step 1: Install
!pip install prompt-amplifier google-generativeai -q
print("✅ Installation complete!")

# Step 2: Set API Key
import os
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"  # Get from https://aistudio.google.com/
print("✅ API key set!")

# Step 3: Import
from prompt_amplifier import PromptForge
from prompt_amplifier.generators import GoogleGenerator
print("✅ Imports complete!")

# Step 4: Create PromptForge
forge = PromptForge(generator=GoogleGenerator())
print("✅ PromptForge created!")

# Step 5: Add knowledge base
forge.add_texts([
    "Company: TechCorp is a B2B SaaS company founded in 2020.",
    "Product: AI-powered analytics platform for sales teams.",
    "Pricing: Starter $99/mo, Professional $299/mo, Enterprise custom pricing.",
    "Support: Email support for Starter, 24/7 chat for Professional, dedicated CSM for Enterprise.",
    "Trial: 14-day free trial available for all plans.",
    "Integration: Connects with Salesforce, HubSpot, and 50+ other tools.",
])
print(f"✅ Added knowledge base! ({forge.chunk_count} chunks)")

# Step 6: Test different prompts
test_prompts = [
    "pricing info",
    "how to get support",
    "product overview",
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"📝 INPUT: {prompt}")
    print(f"{'='*60}")
    
    result = forge.expand(prompt)
    
    print(f"\n📊 Expansion: {result.expansion_ratio:.1f}x")
    print(f"⏱️ Time: {result.total_time_ms:.0f}ms")
    print(f"\n📄 OUTPUT:\n{result.prompt[:800]}...")
```

**Expected Output:**
```
✅ Installation complete!
✅ API key set!
✅ Imports complete!
✅ PromptForge created!
✅ Added knowledge base! (6 chunks)

============================================================
📝 INPUT: pricing info
============================================================

📊 Expansion: 45.2x
⏱️ Time: 2341ms

📄 OUTPUT:
**GOAL:** Provide comprehensive pricing information for TechCorp's products.

**REQUIRED SECTIONS:**

1. **Pricing Tiers Overview**
   | Plan | Price | Key Features |
   |------|-------|--------------|
   | Starter | $99/mo | Basic features |
   | Professional | $299/mo | Advanced features, 24/7 support |
   | Enterprise | Custom | Full features, dedicated CSM |

2. **Feature Comparison**
   - What's included in each tier
   - Upgrade path recommendations
...
```

---

## 📊 Understanding the Result Object

```python
result = forge.expand("Your prompt here")

# Access different parts of the result
print(result.prompt)              # The expanded prompt (string)
print(result.original_prompt)     # Your original input
print(result.expansion_ratio)     # How much longer (e.g., 45.2x)
print(result.retrieval_time_ms)   # Time to search documents (ms)
print(result.generation_time_ms)  # Time to generate prompt (ms)
print(result.total_time_ms)       # Total time (ms)

# Access retrieved context
print(f"Used {len(result.context_chunks)} context chunks")
for chunk in result.context_chunks:
    print(f"  - {chunk.content[:50]}...")
```

**Example Output:**
```
Expanded prompt: **GOAL:** Generate a comprehensive...
Original: pricing info
Expansion ratio: 45.2x
Retrieval time: 12ms
Generation time: 2329ms
Total time: 2341ms
Used 3 context chunks
  - Pricing: Starter $99/mo, Professional $299/mo...
  - Trial: 14-day free trial available for all...
  - Product: AI-powered analytics platform for...
```

---

## 🔧 Troubleshooting

### Error: "API key for OpenAI is missing"

**Problem:** You're trying to use `expand()` without setting an API key.

**Solution:**
```python
import os

# Option 1: Use Google (has free tier)
os.environ["GOOGLE_API_KEY"] = "your-key"
from prompt_amplifier.generators import GoogleGenerator
forge = PromptForge(generator=GoogleGenerator())

# Option 2: Use OpenAI
os.environ["OPENAI_API_KEY"] = "sk-your-key"
forge = PromptForge()  # Uses OpenAI by default
```

### Error: "TF-IDF requires at least 2 documents"

**Problem:** You added only 1 document, but TF-IDF needs 2+.

**Solution:**
```python
# Option 1: Add more documents
forge.add_texts(["doc 1", "doc 2", "doc 3"])

# Option 2: Use Sentence Transformers (works with 1 doc)
from prompt_amplifier.embedders import SentenceTransformerEmbedder
forge = PromptForge(embedder=SentenceTransformerEmbedder())
forge.add_texts(["single document works!"])
```

### Error: "Source not found"

**Problem:** The file/directory path doesn't exist.

**Solution:**
```python
# Check if path exists first
import os
path = "./my_docs/"
if os.path.exists(path):
    forge.load_documents(path)
else:
    print(f"Path not found: {path}")
```

---

## 📚 Key Features

| Feature | Description |
|---------|-------------|
| **10+ Document Loaders** | PDF, DOCX, Excel, CSV, TXT, JSON, Web, YouTube, Sitemap, RSS |
| **12+ Embedders** | TF-IDF, BM25, Sentence Transformers, OpenAI, Google, Cohere, Voyage, Jina, Mistral |
| **3 Vector Stores** | Memory (default), ChromaDB, FAISS |
| **6 LLM Generators** | OpenAI, Anthropic, Google, Ollama, Mistral, Together AI |
| **Evaluation Tools** | Quality metrics, embedder comparison, generator benchmarking |
| **CLI Tool** | Command-line interface for quick testing |

---

## 🎯 What's New in v0.2.1

- ✅ **Input Validation**: Clear errors for empty prompts, missing files
- ✅ **Structured Logging**: Debug visibility with `logging` module
- ✅ **Better Error Messages**: Actionable suggestions in error text
- ✅ **Fixed Generators**: Separate files for Anthropic, Google
- ✅ **Documentation**: Clearer API key requirements

---

## Next Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install Prompt Amplifier and optional dependencies

    [:octicons-arrow-right-24: Installation Guide](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Complete tutorial with examples

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Deep dive into all features

    [:octicons-arrow-right-24: User Guide](guide/concepts.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Detailed API documentation

    [:octicons-arrow-right-24: API Reference](api/promptforge.md)

</div>

---

## Community

- 📦 [PyPI Package](https://pypi.org/project/prompt-amplifier/)
- 🐙 [GitHub Repository](https://github.com/DeccanX/Prompt-Amplifier)
- 🐛 [Issue Tracker](https://github.com/DeccanX/Prompt-Amplifier/issues)
- 📄 [License (Apache 2.0)](https://github.com/DeccanX/Prompt-Amplifier/blob/main/LICENSE)

---

## License

Prompt Amplifier is released under the [Apache 2.0 License](https://github.com/DeccanX/Prompt-Amplifier/blob/main/LICENSE).

Copyright © 2025 Rajesh More and Team
