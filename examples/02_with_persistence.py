"""
PromptForge with Persistent Vector Store

This example demonstrates using ChromaDB for persistent storage,
so embeddings are saved and reused across sessions.

Requirements:
    pip install prompt_amplifier[vectorstore-chroma,embeddings-local]
    export OPENAI_API_KEY="your-key"  # For generation
"""

from pathlib import Path
from prompt_amplifier import PromptForge
from prompt_amplifier.core.config import (
    PromptForgeConfig,
    EmbedderConfig,
    VectorStoreConfig,
    GeneratorConfig,
)

# Configuration with persistence
config = PromptForgeConfig(
    embedder=EmbedderConfig(
        provider="sentence-transformers",
        model="all-MiniLM-L6-v2",  # 384 dimensions, fast
    ),
    vectorstore=VectorStoreConfig(
        provider="chroma",
        collection_name="my_docs",
        persist_directory="./chroma_db",  # Data saved here
    ),
    generator=GeneratorConfig(
        provider="openai",
        model="gpt-4o-mini",
    ),
)

# Initialize PromptForge
forge = PromptForge(config=config)

# Check if we already have data
if forge.chunk_count == 0:
    print("📂 First run: Loading documents...")
    
    # Create sample documents
    docs_dir = Path("./sample_docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Create a sample file
    (docs_dir / "guidelines.txt").write_text("""
    POC Evaluation Guidelines
    ========================
    
    Health Status Definitions:
    - Healthy: All milestones on track, Winscore > 70
    - Warning: Some delays, Winscore 40-70
    - Critical: Major blockers, Winscore < 40
    
    Required Fields:
    - Deal Name
    - Account Executive
    - POC Duration (weeks)
    - Success Criteria
    - Current Status
    """)
    
    (docs_dir / "metrics.txt").write_text("""
    Key Performance Metrics
    ======================
    
    Winscore: Overall confidence score (0-100)
    - Technical fit: 30%
    - Business alignment: 30%
    - Champion strength: 20%
    - Timeline adherence: 20%
    
    Feature Fit: Percentage of required features available
    
    Engagement Score: Customer interaction frequency and quality
    """)
    
    # Load documents
    forge.load_documents(str(docs_dir))
    print(f"✅ Loaded {forge.document_count} documents, {forge.chunk_count} chunks")
    
else:
    print(f"📂 Using existing data: {forge.chunk_count} chunks loaded from ChromaDB")

# Expand a prompt
print("\n" + "=" * 50)
print("Expanding prompt...")
print("=" * 50)

result = forge.expand("Give me a deal health summary")

print(f"\n🎯 Expanded Prompt:\n")
print(result.prompt)
print(f"\n📊 Stats:")
print(f"  - Context chunks used: {result.context_count}")
print(f"  - Retrieval time: {result.retrieval_time_ms:.0f}ms")
print(f"  - Generation time: {result.generation_time_ms:.0f}ms")
print(f"  - Expansion ratio: {result.expansion_ratio:.1f}x")

