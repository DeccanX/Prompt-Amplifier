# Contributing

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

# Install additional test dependencies
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

1. Check [existing issues](https://github.com/DeccanX/Prompt-Amplifier/issues)
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

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** following our coding standards

3. **Add tests** for new functionality

4. **Ensure all tests pass:**
   ```bash
   pytest tests/ -v
   ```

5. **Format and lint:**
   ```bash
   black src/ tests/
   ruff check src/ tests/
   ```

6. **Commit with clear messages:**
   ```bash
   git commit -m "feat: add new embedder for XYZ"
   ```

7. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Description |
|--------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation |
| `test:` | Adding tests |
| `refactor:` | Code refactoring |
| `style:` | Formatting |
| `chore:` | Maintenance |

Examples:
```
feat: add Cohere embedder support
fix: handle empty document list in chunker
docs: add tutorial for research assistant
test: add tests for hybrid retriever
```

## Code Style

### Python Style

- Use [Black](https://black.readthedocs.io/) (line length: 100)
- Follow [PEP 8](https://pep8.org/)
- Add type hints to all functions
- Write docstrings for public APIs

### Example

```python
from __future__ import annotations


def embed(self, texts: list[str]) -> EmbeddingResult:
    """
    Embed a list of texts into vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        EmbeddingResult containing embeddings and metadata.

    Raises:
        EmbedderError: If embedding fails.
    
    Example:
        >>> embedder = OpenAIEmbedder()
        >>> result = embedder.embed(["Hello", "World"])
        >>> print(result.dimension)
        1536
    """
    ...
```

## Adding New Components

### New Embedder

1. Create `src/prompt_amplifier/embedders/your_embedder.py`:

```python
from __future__ import annotations

from prompt_amplifier.embedders.base import BaseEmbedder
from prompt_amplifier.models import EmbeddingResult


class YourEmbedder(BaseEmbedder):
    """Your embedder description."""
    
    def __init__(self, model: str = "default-model"):
        self.model = model
    
    @property
    def dimension(self) -> int:
        return 768
    
    def embed(self, texts: list[str]) -> EmbeddingResult:
        # Implementation
        ...
```

2. Add to `src/prompt_amplifier/embedders/__init__.py`
3. Add tests in `tests/test_embedders.py`
4. Update documentation

### New Vector Store

1. Create `src/prompt_amplifier/vectorstores/your_store.py`
2. Inherit from `BaseVectorStore`
3. Implement: `add()`, `search()`, `delete()`, `count`
4. Add tests and documentation

### New Generator

1. Create `src/prompt_amplifier/generators/your_generator.py`
2. Inherit from `BaseGenerator`
3. Implement: `generate()`
4. Add tests and documentation

## Testing Guidelines

### Write Tests

- Unit tests for all new code
- Use pytest fixtures
- Mock external API calls
- Aim for >80% coverage

### Test Structure

```python
import pytest
from prompt_amplifier.embedders import YourEmbedder


class TestYourEmbedder:
    """Tests for YourEmbedder."""
    
    @pytest.fixture
    def embedder(self):
        return YourEmbedder()
    
    def test_embed_basic(self, embedder):
        """Test basic embedding."""
        result = embedder.embed(["test"])
        assert result.count == 1
        assert result.dimension > 0
    
    def test_embed_empty(self, embedder):
        """Test empty input."""
        result = embedder.embed([])
        assert result.count == 0
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def function(arg1: str, arg2: int = 10) -> bool:
    """
    Short description.

    Longer description if needed.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2. Defaults to 10.

    Returns:
        Description of return value.

    Raises:
        ValueError: When arg1 is empty.
    
    Example:
        >>> function("test", 20)
        True
    """
```

### Update Docs

When adding features, update:

- API reference in `docs/api/`
- Guides in `docs/guide/`
- Examples if relevant

## Review Process

1. All PRs require review
2. CI must pass (tests, lint, format)
3. Documentation must be updated
4. Breaking changes need discussion

## Questions?

- Open an issue for questions
- Email: moreyrb@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for contributing! 🙏

