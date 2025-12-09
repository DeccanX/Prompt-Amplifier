# Contributing to Prompt Amplifier

Thank you for your interest in contributing to Prompt Amplifier! 🎉

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/Prompt-Amplifier.git
cd Prompt-Amplifier
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install additional dependencies for testing
pip install sentence-transformers rank-bm25 chromadb
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=prompt_amplifier --cov-report=html

# Run specific test file
pytest tests/test_embedders.py -v
```

### 4. Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## How to Contribute

### Reporting Bugs

1. Check existing [issues](https://github.com/DeccanX/Prompt-Amplifier/issues)
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS

### Suggesting Features

1. Open an issue with the "enhancement" label
2. Describe the feature and use case
3. Provide example code if possible

### Pull Requests

1. Create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Add tests for new functionality

4. Ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```

5. Format and lint your code:
   ```bash
   black src/ tests/
   ruff check src/ tests/
   ```

6. Commit with clear messages:
   ```bash
   git commit -m "feat: add new embedder for XYZ"
   ```

7. Push and create a PR:
   ```bash
   git push origin feature/your-feature-name
   ```

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding/updating tests
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

## Code Style

- Use [Black](https://black.readthedocs.io/) for formatting (line length: 100)
- Follow [PEP 8](https://pep8.org/) guidelines
- Add type hints to all functions
- Write docstrings for public APIs

### Example

```python
def embed(self, texts: list[str]) -> EmbeddingResult:
    """
    Embed a list of texts into vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        EmbeddingResult containing embeddings and metadata.

    Raises:
        EmbedderError: If embedding fails.
    """
    ...
```

## Adding New Components

### New Embedder

1. Create `src/prompt_amplifier/embedders/your_embedder.py`
2. Inherit from `BaseEmbedder` or `BaseSparseEmbedder`
3. Implement required methods: `embed()`, `dimension`
4. Add to `embedders/__init__.py`
5. Add tests in `tests/test_embedders.py`

### New Vector Store

1. Create `src/prompt_amplifier/vectorstores/your_store.py`
2. Inherit from `BaseVectorStore`
3. Implement required methods: `add()`, `search()`, `count`
4. Add to `vectorstores/__init__.py`
5. Add tests in `tests/test_vectorstores.py`

### New Generator

1. Create `src/prompt_amplifier/generators/your_generator.py`
2. Inherit from `BaseGenerator`
3. Implement required methods: `generate()`
4. Add to `generators/__init__.py`
5. Add tests in `tests/test_generators.py`

## Testing Guidelines

- Write unit tests for all new code
- Use pytest fixtures for common setup
- Mock external API calls
- Aim for >80% coverage on new code

```python
import pytest
from prompt_amplifier.embedders import YourEmbedder

class TestYourEmbedder:
    def test_embed_basic(self):
        embedder = YourEmbedder()
        result = embedder.embed(["test text"])
        assert result.count == 1
        assert result.dimension > 0
```

## Questions?

- Open an issue for questions
- Email: moreyrb@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for contributing! 🙏

