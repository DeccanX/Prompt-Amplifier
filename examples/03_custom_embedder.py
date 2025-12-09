"""
PromptForge with Custom Embedder

This example shows how to use different embedding providers:
- TF-IDF (free, local, sparse)
- Sentence Transformers (free, local, dense)
- OpenAI (paid, API)

Requirements:
    pip install prompt_amplifier[all]
"""

from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import TFIDFEmbedder
from prompt_amplifier.vectorstores import MemoryStore

# Sample corpus
corpus = [
    "The quarterly sales report shows a 15% increase in revenue.",
    "Customer satisfaction scores improved to 4.5 out of 5 stars.",
    "The new product launch exceeded expectations with 10,000 pre-orders.",
    "Technical support tickets decreased by 20% this quarter.",
    "Employee retention rate improved to 95% after new benefits package.",
    "Market share increased from 12% to 15% in the enterprise segment.",
]


def demo_tfidf():
    """Demo with TF-IDF embedder (sparse, free, fast)."""
    print("\n" + "=" * 50)
    print("🔤 TF-IDF Embedder Demo")
    print("=" * 50)

    # TF-IDF needs to be fitted on corpus first
    embedder = TFIDFEmbedder(max_features=1000, ngram_range=(1, 2))

    # Create forge with custom embedder
    forge = PromptForge(embedder=embedder)
    forge.add_texts(corpus, source="business_reports")

    # The TF-IDF embedder is automatically fitted when chunks are embedded
    print(f"Vocabulary size: {embedder.vocabulary_size}")
    print(f"Embedding dimension: {embedder.dimension}")

    # Search
    results = forge.search("revenue sales growth", top_k=3)
    print("\nSearch results for 'revenue sales growth':")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.score:.3f}] {r.content[:60]}...")


def demo_sentence_transformers():
    """Demo with Sentence Transformers (dense, free, local)."""
    print("\n" + "=" * 50)
    print("🤖 Sentence Transformers Demo")
    print("=" * 50)

    try:
        from prompt_amplifier.embedders import SentenceTransformerEmbedder
    except ImportError:
        print("Install sentence-transformers: pip install sentence-transformers")
        return

    embedder = SentenceTransformerEmbedder(
        model="all-MiniLM-L6-v2",
        normalize_embeddings=True,
    )

    forge = PromptForge(embedder=embedder)
    forge.add_texts(corpus, source="business_reports")

    print(f"Model: {embedder.model}")
    print(f"Embedding dimension: {embedder.dimension}")

    # Search - semantic understanding
    results = forge.search("how are customers feeling?", top_k=3)
    print("\nSearch results for 'how are customers feeling?':")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.score:.3f}] {r.content[:60]}...")


def demo_openai():
    """Demo with OpenAI embeddings (dense, paid, API)."""
    print("\n" + "=" * 50)
    print("🌐 OpenAI Embeddings Demo")
    print("=" * 50)

    try:
        from prompt_amplifier.embedders import OpenAIEmbedder
    except ImportError:
        print("Install openai: pip install openai")
        return

    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable")
        return

    embedder = OpenAIEmbedder(
        model="text-embedding-3-small",
        # dimensions=512,  # Optional: reduce dimensions for cost savings
    )

    forge = PromptForge(embedder=embedder)
    forge.add_texts(corpus, source="business_reports")

    print(f"Model: {embedder.model}")
    print(f"Embedding dimension: {embedder.dimension}")

    # Search
    results = forge.search("company performance metrics", top_k=3)
    print("\nSearch results for 'company performance metrics':")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.score:.3f}] {r.content[:60]}...")


if __name__ == "__main__":
    print("=" * 50)
    print("PromptForge Custom Embedder Examples")
    print("=" * 50)

    # Always works (no external deps for TF-IDF)
    demo_tfidf()

    # Requires sentence-transformers
    demo_sentence_transformers()

    # Requires openai + API key
    demo_openai()
