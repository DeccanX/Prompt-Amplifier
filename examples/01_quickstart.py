"""
PromptForge Quickstart Example

This example shows the basic usage of PromptForge to transform
short prompts into detailed, structured instructions.

Requirements:
    pip install prompt_amplifier

For full expansion (with LLM):
    export OPENAI_API_KEY="your-key"
"""

import os
from prompt_amplifier import PromptForge

# Initialize with defaults (TF-IDF embedder, in-memory store)
forge = PromptForge()

# Add some sample documents
sample_docs = [
    """
    POC Health Check Guidelines:
    - Healthy: All milestones on track, positive customer engagement
    - Warning: 1-2 milestones delayed, some concerns raised
    - Critical: Multiple blockers, customer disengaged, timeline at risk
    
    Key metrics to evaluate:
    - Winscore (0-100): Overall deal confidence
    - Feature fit percentage
    - Customer engagement score
    - Days since last activity
    """,
    """
    Success Plan Fields:
    - Deal Name
    - Account Executive
    - Technical Lead
    - POC Start Date
    - POC End Date
    - Success Criteria (list)
    - Milestones (with dates)
    - Blockers and Risks
    - Next Steps
    """,
    """
    Deal Stages:
    1. Discovery - Understanding customer needs
    2. Technical Validation - POC/Pilot phase
    3. Business Validation - ROI and stakeholder alignment
    4. Negotiation - Contract and pricing
    5. Closed Won/Lost
    """,
]

# Add texts directly
forge.add_texts(sample_docs, source="knowledge_base")

print("=" * 60)
print("PromptForge Quickstart Demo")
print("=" * 60)
print(f"\n📊 Loaded {forge.document_count} documents, {forge.chunk_count} chunks")

# Check if OpenAI API key is available
HAS_API_KEY = bool(os.getenv("OPENAI_API_KEY"))

if HAS_API_KEY:
    # Example 1: POC Health Check (requires LLM)
    print("\n📝 Example 1: POC Health Check")
    print("-" * 40)

    short_prompt = "How's the deal going?"
    result = forge.expand(short_prompt)

    print(f"Original: {short_prompt}")
    print(f"\nExpanded ({result.expansion_ratio:.1f}x longer):")
    print(result.prompt)
    print(f"\nContext sources used: {result.context_count}")
    print(f"Generation time: {result.total_time_ms:.0f}ms")

    # Example 2: Feature Fit Analysis
    print("\n\n📝 Example 2: Feature Fit Analysis")
    print("-" * 40)

    short_prompt = "Analyze feature fit"
    result = forge.expand(short_prompt)

    print(f"Original: {short_prompt}")
    print(f"\nExpanded ({result.expansion_ratio:.1f}x longer):")
    print(result.prompt)
else:
    print("\n⚠️  OPENAI_API_KEY not set - skipping LLM expansion examples")
    print("   Set the key to see full prompt expansion in action:")
    print("   export OPENAI_API_KEY='your-key'")

# Example 3: Search only (no expansion needed, works without API key)
print("\n\n🔍 Search Demo (no API key required)")
print("-" * 40)

queries = [
    "success criteria milestones",
    "deal health status",
    "customer engagement",
]

for query in queries:
    search_results = forge.search(query, top_k=2)
    print(f"\nQuery: '{query}'")

    for i, r in enumerate(search_results, 1):
        preview = r.content[:80].replace("\n", " ").strip()
        print(f"  {i}. [{r.score:.3f}] {preview}...")

print("\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
