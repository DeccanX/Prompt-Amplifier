# Prompt Amplifier 🔨

**Transform short prompts into detailed, structured instructions using context-aware retrieval.**

[![PyPI version](https://badge.fury.io/py/prompt-amplifier.svg)](https://pypi.org/project/prompt-amplifier/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## What is Prompt Amplifier?

Prompt Amplifier is a library for **Prompt Amplification** — the process of transforming short, ambiguous user intents into comprehensive, well-structured prompts that LLMs can execute effectively.

## Quick Example

```python
from prompt_amplifier import PromptForge

forge = PromptForge()
forge.add_texts(["Your knowledge base content..."])

# Short input
result = forge.expand("How's the deal going?")

# Detailed output
print(result.prompt)
```

**Before:** `"How's the deal going?"`

**After:** A detailed, structured prompt with sections, tables, and specific instructions!

## Features

- 📄 **Multi-format Document Loading** — PDF, DOCX, Excel, CSV, TXT, JSON
- 🔢 **Pluggable Embedders** — TF-IDF, BM25, Sentence Transformers, OpenAI, Google
- 💾 **Vector Store Support** — In-memory, ChromaDB, FAISS
- 🔍 **Smart Retrieval** — Vector search, hybrid (BM25 + Vector)
- 🤖 **LLM Backends** — OpenAI, Anthropic, Google Gemini

## Installation

```bash
pip install prompt-amplifier
```

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [API Reference](api/promptforge.md)

