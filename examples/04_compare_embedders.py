"""
Compare TF-IDF vs Dense Embeddings

This example shows the difference between:
- TF-IDF (keyword matching, sparse)
- Sentence Transformers (semantic understanding, dense)
"""

# Sample corpus
corpus = [
    """POC Health Check Guidelines:
    - Healthy: All milestones on track, positive customer engagement
    - Warning: 1-2 milestones delayed, some concerns raised
    - Critical: Multiple blockers, customer disengaged, timeline at risk
    
    Key metrics: Winscore (0-100), Feature fit %, Customer engagement score""",
    
    """Success Plan Fields:
    - Deal Name, Account Executive, Technical Lead
    - POC Start/End Date, Success Criteria, Milestones
    - Blockers and Risks, Next Steps""",
    
    """Deal Stages:
    1. Discovery - Understanding customer needs
    2. Technical Validation - POC/Pilot phase
    3. Business Validation - ROI alignment
    4. Negotiation - Contract and pricing
    5. Closed Won/Lost""",
]

# Test queries - some exact match, some semantic
test_queries = [
    ("success criteria milestones", "Exact keywords in corpus"),
    ("deal health status", "Partial keyword match"),
    ("how is the project going?", "Semantic - no exact keywords"),
    ("are we on track?", "Semantic - no exact keywords"),
    ("customer happiness", "Semantic - similar to 'engagement'"),
]

print("=" * 70)
print("Embedder Comparison: TF-IDF vs Sentence Transformers")
print("=" * 70)

# === TF-IDF ===
print("\n🔤 TF-IDF (Keyword-based, Sparse)")
print("-" * 50)

from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import TFIDFEmbedder

forge_tfidf = PromptForge(embedder=TFIDFEmbedder())
forge_tfidf.add_texts(corpus, source="docs")

for query, description in test_queries:
    results = forge_tfidf.search(query, top_k=1)
    if results.results:
        score = results.results[0].score
        preview = results.results[0].content[:50].replace('\n', ' ')
        print(f"  '{query}' → [{score:.3f}] {preview}...")
    else:
        print(f"  '{query}' → No results")

# === Sentence Transformers ===
print("\n🤖 Sentence Transformers (Semantic, Dense)")
print("-" * 50)

try:
    from prompt_amplifier.embedders import SentenceTransformerEmbedder
    
    forge_st = PromptForge(embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"))
    forge_st.add_texts(corpus, source="docs")
    
    for query, description in test_queries:
        results = forge_st.search(query, top_k=1)
        if results.results:
            score = results.results[0].score
            preview = results.results[0].content[:50].replace('\n', ' ')
            print(f"  '{query}' → [{score:.3f}] {preview}...")
        else:
            print(f"  '{query}' → No results")

except ImportError:
    print("  ⚠️  Install sentence-transformers: pip install sentence-transformers")

# === Summary ===
print("\n" + "=" * 70)
print("📊 Summary")
print("=" * 70)
print("""
| Query Type           | TF-IDF        | Sentence Transformers |
|---------------------|---------------|----------------------|
| Exact keywords      | ✅ Good       | ✅ Good              |
| Partial keywords    | ⚠️ Weak       | ✅ Good              |
| Semantic meaning    | ❌ Poor       | ✅ Excellent         |
| Speed               | ⚡ Very fast  | 🐢 Slower            |
| Memory              | 📦 Low        | 📦 Higher            |
| API cost            | 💰 Free       | 💰 Free (local)      |

Recommendation:
- Small corpus + exact keywords → TF-IDF
- Large corpus + semantic search → Sentence Transformers or OpenAI
- Best of both → Hybrid search (BM25 + Dense)
""")

