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

## Quick Example

```python
from prompt_amplifier import PromptForge

# Initialize with your documents
forge = PromptForge()
forge.load_documents("./sales_data/")

# Transform a simple prompt
result = forge.expand("How's the deal going?")

print(result.prompt)
```

**Output:**

```markdown
**GOAL:** Provide a comprehensive analysis of the current deal health status.

**CONTEXT:**
Based on the POC tracking data, analyze:
- Current POC health status (Healthy/At Risk/Critical)
- Milestone completion percentage
- Winscore trends
- Feature fit analysis

**SECTIONS:**
1. Executive Summary
2. POC Health Assessment
3. Key Metrics Analysis
4. Risk Factors
5. Recommendations

**INSTRUCTIONS:**
- Use specific data points from the context
- Include quantitative metrics where available
- Highlight any concerns or blockers
- Format numbers consistently
```

---

## Key Features

### 📥 Multi-Format Document Loading
Load data from 10+ formats including PDF, DOCX, CSV, JSON, Web Pages, YouTube, RSS feeds, and more.

### 🔢 Flexible Embedding Strategies
Choose from 12+ embedding providers:
- **Free/Local**: TF-IDF, BM25, Sentence Transformers, FastEmbed
- **Cloud APIs**: OpenAI, Google, Cohere, Voyage, Jina, Mistral

### 💾 Vector Store Integration
Persist your embeddings with ChromaDB, FAISS, or in-memory storage.

### 🤖 Multiple LLM Backends
Generate with OpenAI, Anthropic, Google, Ollama (local), Mistral, or Together AI.

### 📊 Built-in Evaluation
Measure prompt quality, compare embedders, and benchmark generators.

### 💻 CLI Tool
Expand prompts directly from the command line.

---

## Installation

```bash
# Basic installation
pip install prompt-amplifier

# With all features
pip install prompt-amplifier[all]

# Specific features
pip install prompt-amplifier[embeddings-openai,generators-anthropic]
```

---

## Use Cases

### 🎯 Sales Intelligence
Transform "How's the deal?" into comprehensive deal health reports.

### 📚 Research Assistant
Convert "Summarize this paper" into structured research summaries.

### 🎧 Customer Support
Turn "Help with billing" into detailed troubleshooting guides.

### 📝 Content Creation
Expand "Write about AI" into well-researched, structured articles.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        PromptForge                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌───────────┐  │
│  │ Loaders │ → │Chunkers │ → │Embedders │ → │VectorStore│  │
│  └─────────┘   └─────────┘   └──────────┘   └───────────┘  │
│       ↓                                           ↓        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Retriever                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Generator                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                │
│                    Expanded Prompt                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Supported Integrations

| Category | Integrations |
|----------|-------------|
| **Loaders** | TXT, CSV, JSON, DOCX, Excel, PDF, Web, YouTube, Sitemap, RSS |
| **Embedders** | TF-IDF, BM25, Sentence Transformers, FastEmbed, OpenAI, Google, Cohere, Voyage, Jina, Mistral |
| **Vector Stores** | Memory, ChromaDB, FAISS |
| **Generators** | OpenAI, Anthropic, Google, Ollama, Mistral, Together AI |

---

## What's New in v0.2.0

- 🌐 **Web Loaders**: WebLoader, YouTubeLoader, SitemapLoader, RSSLoader
- 🔢 **New Embedders**: Cohere, Voyage, Jina, Mistral
- 🤖 **New Generators**: Ollama, Mistral, Together AI
- 📊 **Evaluation Module**: Quality metrics, benchmarking, test suites
- 💻 **CLI Tool**: Command-line interface

[See full changelog →](changelog.md)

---

## Getting Started

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install Prompt Amplifier and its dependencies

    [:octicons-arrow-right-24: Installation Guide](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Get up and running in 5 minutes

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

Copyright © 2024 Rajesh More
