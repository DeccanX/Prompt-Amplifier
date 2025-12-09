# Changelog

All notable changes to Prompt Amplifier are documented here.

---

## [0.2.0] - 2024-12-09

### 🚀 New Features

#### Web Loaders
- **WebLoader**: Fetch and parse web pages with BeautifulSoup
- **YouTubeLoader**: Extract video transcripts using `youtube-transcript-api`
- **SitemapLoader**: Crawl entire websites via sitemap.xml
- **RSSLoader**: Load content from RSS/Atom feeds

#### New Embedders
- **CohereEmbedder**: Cohere embedding API integration
- **CohereRerankEmbedder**: Cohere reranking for improved retrieval
- **VoyageEmbedder**: Voyage AI embeddings for specialized domains
- **JinaEmbedder**: Jina AI embeddings with multilingual support
- **MistralEmbedder**: Mistral AI embeddings (EU-based)

#### New Generators
- **OllamaGenerator**: Run LLMs locally with no API costs
- **MistralGenerator**: Mistral AI API for European compliance
- **TogetherGenerator**: Access 50+ open-source models via Together AI

#### Evaluation Module
- `calculate_expansion_quality()`: Measure prompt quality (structure, specificity, completeness, readability)
- `calculate_retrieval_metrics()`: Standard IR metrics (precision, recall, MRR, NDCG)
- `calculate_diversity_score()`: Measure result diversity
- `calculate_coherence_score()`: Measure prompt-context coherence
- `compare_embedders()`: Benchmark multiple embedders
- `benchmark_generators()`: Benchmark multiple LLMs
- `EvaluationSuite`: Run comprehensive test suites

#### CLI Tool
- `prompt-amplifier expand`: Expand prompts from command line
- `prompt-amplifier search`: Search documents
- `prompt-amplifier compare-embedders`: Benchmark embedders
- `prompt-amplifier evaluate`: Run evaluation suite
- `prompt-amplifier version`: Show version

### 📦 Dependencies

New optional dependency groups:
- `[loaders]`: beautifulsoup4, youtube-transcript-api, feedparser
- `[embeddings-cohere]`: cohere
- `[embeddings-voyage]`: voyageai
- `[embeddings-jina]`: jina-embeddings
- `[embeddings-mistral]`: mistralai
- `[generators-ollama]`: ollama
- `[generators-mistral]`: mistralai
- `[generators-together]`: together

### 📚 Documentation

- Added Web Loaders guide
- Added Evaluation Metrics guide
- Added Additional Embedders guide
- Added Additional Generators guide
- Added CLI documentation
- Updated API reference

---

## [0.1.5] - 2024-12-08

### 🐛 Bug Fixes

- Fixed CI/CD pipeline for all Python versions (3.9-3.12)
- Fixed type hints compatibility for Python 3.9/3.10
- Fixed Black formatting issues
- Optimized GitHub Actions disk space usage

### 📦 Dependencies

- Updated `numpy>=1.24.0` for Python 3.12 compatibility
- Updated `scikit-learn>=1.3.0` for Python 3.12 compatibility

---

## [0.1.4] - 2024-12-08

### ✨ Improvements

- Added `from __future__ import annotations` for Python 3.9/3.10 compatibility
- Fixed Ruff linting configuration

---

## [0.1.3] - 2024-12-08

### 🔧 Changes

- Renamed package from `promptforge` to `prompt-amplifier`
- Updated all import paths
- Updated documentation

---

## [0.1.2] - 2024-12-07

### ✨ Initial Release

Core features:
- **PromptForge Engine**: Main orchestration class
- **Document Loaders**: TXT, CSV, JSON, DOCX, Excel, PDF
- **Chunkers**: RecursiveChunker with configurable size/overlap
- **Embedders**: TF-IDF, BM25, Sentence Transformers, OpenAI, Google
- **Vector Stores**: Memory, ChromaDB, FAISS
- **Retrievers**: Vector, Hybrid
- **Generators**: OpenAI, Anthropic, Google

---

## Roadmap

### v0.3.0 (Planned)

- [ ] More vector store integrations (Pinecone, Qdrant, Weaviate)
- [ ] Streaming support for generators
- [ ] Async API support
- [ ] Custom prompt templates
- [ ] Multi-modal support (images)

### v1.0.0 (Future)

- [ ] Stable API
- [ ] Production-ready performance
- [ ] Enterprise features
- [ ] Comprehensive benchmarks

---

## Contributing

See [Contributing Guide](contributing.md) for how to submit changes.

Report bugs and feature requests on [GitHub Issues](https://github.com/DeccanX/Prompt-Amplifier/issues).
