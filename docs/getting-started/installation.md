# Installation

## Basic Installation

```bash
pip install prompt-amplifier
```

This installs the core library with TF-IDF embeddings.

## Optional Dependencies

### Local Embeddings (Free)

```bash
pip install prompt-amplifier[embeddings-local]
```

Includes Sentence Transformers for semantic embeddings.

### OpenAI Integration

```bash
pip install prompt-amplifier[embeddings-openai,generators-openai]
```

### All Features

```bash
pip install prompt-amplifier[all]
```

Includes all embedders, vector stores, and generators.

## Development Installation

```bash
git clone https://github.com/DeccanX/Prompt-Amplifier.git
cd Prompt-Amplifier
pip install -e ".[dev]"
```

## Requirements

- Python 3.9 or higher
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- pydantic >= 2.0.0

