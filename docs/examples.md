# Examples

## Basic Usage

```python
from prompt_amplifier import PromptForge

forge = PromptForge()
forge.add_texts([
    "POC Health: Healthy = on track, Warning = delays, Critical = blocked",
    "Success metrics: Winscore 0-100, Feature fit %, Engagement score",
])

result = forge.expand("Check deal health")
print(result.prompt)
```

## With ChromaDB Persistence

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.vectorstores import ChromaStore
from prompt_amplifier.embedders import SentenceTransformerEmbedder

forge = PromptForge(
    embedder=SentenceTransformerEmbedder(),
    vectorstore=ChromaStore(
        collection_name="sales_docs",
        persist_directory="./db"
    )
)

# First run: load and embed
forge.load_documents("./docs/")

# Later: reuse existing embeddings
result = forge.expand("Summarize project")
```

## Compare Embedders

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import TFIDFEmbedder, SentenceTransformerEmbedder

texts = ["Sales increased by 15%", "Customer satisfaction at 4.5 stars"]

# TF-IDF (keyword-based)
forge_tfidf = PromptForge(embedder=TFIDFEmbedder())
forge_tfidf.add_texts(texts)

# Sentence Transformers (semantic)
forge_st = PromptForge(embedder=SentenceTransformerEmbedder())
forge_st.add_texts(texts)

# Compare search results
print("TF-IDF:", forge_tfidf.search("revenue growth"))
print("Semantic:", forge_st.search("revenue growth"))
```

## Multi-Provider LLM

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.core.config import PromptForgeConfig, GeneratorConfig

providers = ["openai", "anthropic", "google"]

for provider in providers:
    config = PromptForgeConfig(
        generator=GeneratorConfig(provider=provider)
    )
    forge = PromptForge(config=config)
    forge.add_texts(["Your knowledge base..."])
    
    result = forge.expand("Summarize")
    print(f"{provider}: {result.expansion_ratio:.1f}x")
```

