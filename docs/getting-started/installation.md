# Installation

## Requirements

- Python 3.9 or higher
- pip (Python package manager)

## Basic Installation

Install the core library with TF-IDF embeddings:

```bash
pip install prompt-amplifier
```

This gives you:

- ✅ TF-IDF embeddings (free, local)
- ✅ BM25 embeddings (free, local)
- ✅ In-memory vector store
- ✅ All document loaders

## Installation Options

### With Sentence Transformers (Recommended)

For semantic embeddings that run locally:

```bash
pip install prompt-amplifier[embeddings-local]
```

### With OpenAI

For OpenAI embeddings and GPT generators:

```bash
pip install prompt-amplifier[embeddings-openai,generators-openai]
```

### With ChromaDB

For persistent local vector storage:

```bash
pip install prompt-amplifier[vectorstore-chroma]
```

### With Everything

Install all features:

```bash
pip install prompt-amplifier[all]
```

This includes:

- All embedders (TF-IDF, BM25, Sentence Transformers, OpenAI, Google)
- All vector stores (Memory, ChromaDB, FAISS)
- All generators (OpenAI, Anthropic, Google)
- All document loaders

## Installation Groups

| Group | Command | Includes |
|-------|---------|----------|
| Core | `pip install prompt-amplifier` | TF-IDF, Memory store |
| Local AI | `pip install prompt-amplifier[embeddings-local]` | + Sentence Transformers |
| OpenAI | `pip install prompt-amplifier[embeddings-openai,generators-openai]` | + OpenAI embeddings & GPT |
| Anthropic | `pip install prompt-amplifier[generators-anthropic]` | + Claude |
| Google | `pip install prompt-amplifier[embeddings-google,generators-google]` | + Gemini |
| ChromaDB | `pip install prompt-amplifier[vectorstore-chroma]` | + ChromaDB |
| FAISS | `pip install prompt-amplifier[vectorstore-faiss]` | + FAISS |
| All | `pip install prompt-amplifier[all]` | Everything |
| Dev | `pip install prompt-amplifier[dev]` | + pytest, black, ruff |

## Development Installation

For contributing to Prompt Amplifier:

```bash
# Clone the repository
git clone https://github.com/DeccanX/Prompt-Amplifier.git
cd Prompt-Amplifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install additional dependencies
pip install sentence-transformers chromadb

# Run tests
pytest tests/ -v
```

## Verify Installation

```python
import prompt_amplifier
print(f"Prompt Amplifier v{prompt_amplifier.__version__}")

from prompt_amplifier import PromptForge
forge = PromptForge()
print("✅ Installation successful!")
```

## Environment Variables

Set API keys for cloud services:

=== "Linux/macOS"

    ```bash
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export GOOGLE_API_KEY="AIza..."
    ```

=== "Windows"

    ```cmd
    set OPENAI_API_KEY=sk-...
    set ANTHROPIC_API_KEY=sk-ant-...
    set GOOGLE_API_KEY=AIza...
    ```

=== ".env file"

    ```bash
    # .env
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    GOOGLE_API_KEY=AIza...
    ```

## Troubleshooting

### ImportError: No module named 'prompt_amplifier'

Make sure you installed the package:

```bash
pip install prompt-amplifier
```

### sentence-transformers not found

Install the local embeddings extra:

```bash
pip install prompt-amplifier[embeddings-local]
```

### chromadb not found

Install the ChromaDB extra:

```bash
pip install prompt-amplifier[vectorstore-chroma]
```

### API key errors

Set the appropriate environment variable:

```bash
export OPENAI_API_KEY="your-key-here"
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started in 5 minutes
- [Configuration](configuration.md) - Learn about configuration options
