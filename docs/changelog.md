# Changelog

All notable changes to Prompt Amplifier are documented here.

## [0.1.4] - 2024-12-09

### Added
- GitHub Actions CI/CD pipeline
- MkDocs documentation site
- Comprehensive test suite (104 tests)
- Python 3.9, 3.10, 3.11, 3.12 support

### Fixed
- Python 3.12 compatibility issues
- Black formatting compliance
- Disk space issues in CI

### Changed
- Updated numpy to >=1.24.0
- Updated scikit-learn to >=1.3.0

## [0.1.1] - 2024-12-09

### Added
- Initial public release
- Core PromptForge engine
- Document loaders (TXT, CSV, JSON, DOCX, Excel, PDF)
- Embedders (TF-IDF, BM25, Sentence Transformers, OpenAI, Google)
- Vector stores (Memory, ChromaDB, FAISS)
- LLM generators (OpenAI, Anthropic, Google Gemini)
- Retrieval strategies (Vector, Hybrid)
- Example scripts

### Documentation
- README with quick start guide
- API design document
- Example usage code

## [0.1.0] - 2024-12-09

### Added
- Initial development release
- Basic project structure
- Core functionality

---

## Version Format

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Upcoming

### Planned for v0.2.0

- [ ] CLI tool (`prompt-amplifier expand "query"`)
- [ ] Cohere embedder
- [ ] Pinecone vector store
- [ ] Qdrant vector store
- [ ] Ollama generator (local LLMs)
- [ ] Streaming support
- [ ] Async API

### Planned for v0.3.0

- [ ] Multi-language support
- [ ] Custom prompt templates
- [ ] Evaluation metrics
- [ ] Benchmark suite
- [ ] Plugin system

