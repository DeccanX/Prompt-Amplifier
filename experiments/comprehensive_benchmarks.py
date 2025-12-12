"""
Comprehensive benchmark experiments for PRIME research paper.
Covers multi-domain evaluation, ablation studies, and caching performance.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

# Set API keys
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# Domain-specific test data
DOMAIN_DATA = {
    "sales": {
        "documents": [
            "POC Health Status: Healthy means all milestones are on track with positive customer engagement.",
            "Warning status indicates delays or risks that need immediate attention from the account team.",
            "Critical status means major blockers that require executive escalation within 24 hours.",
            "Winscore ranges from 0-100, measuring deal probability based on multiple factors.",
            "Feature Fit shows product-customer alignment as a percentage of required features.",
            "Deal Velocity measures how fast a deal progresses through sales stages.",
            "Executive Sponsor presence increases close rate by 40% on average.",
            "Renewal Probability is calculated from usage metrics and support ticket sentiment.",
        ],
        "queries": [
            "How's the deal going?",
            "Check pipeline health",
            "What's the forecast?",
            "Analyze deal risks",
        ],
    },
    "research": {
        "documents": [
            "Abstract: Summarizes the main findings, methodology, and contributions of the paper.",
            "Introduction: Presents the research problem, motivation, and paper organization.",
            "Related Work: Reviews existing literature and positions the contribution.",
            "Methodology: Describes the approach, algorithms, and experimental setup.",
            "Results: Presents quantitative findings with statistical significance tests.",
            "Discussion: Interprets results, acknowledges limitations, suggests future work.",
            "Citation format: Author (Year) for in-text, full reference in bibliography.",
            "Peer review process: Double-blind review with 2-3 expert reviewers.",
        ],
        "queries": [
            "Summarize the paper",
            "What's the methodology?",
            "Key contributions",
            "Literature review",
        ],
    },
    "customer_support": {
        "documents": [
            "Tier 1 Support: Basic troubleshooting, password resets, FAQ responses.",
            "Tier 2 Support: Technical issues requiring product expertise and logs analysis.",
            "Tier 3 Support: Engineering escalation for bugs and feature requests.",
            "SLA: Premium customers get 4-hour response, Standard gets 24-hour response.",
            "Ticket Priority: P1 (critical), P2 (high), P3 (medium), P4 (low).",
            "CSAT Score: Customer satisfaction measured post-resolution on 1-5 scale.",
            "First Response Time: Average time from ticket creation to first agent response.",
            "Resolution Time: Average time from ticket creation to marked resolved.",
        ],
        "queries": [
            "Help with billing",
            "Product not working",
            "How to upgrade",
            "Cancel subscription",
        ],
    },
    "content_creation": {
        "documents": [
            "Blog posts should be 1500-2500 words for optimal SEO performance.",
            "Use H2 and H3 headers to structure content for readability.",
            "Include internal links to related content and external links to sources.",
            "Meta description should be 150-160 characters summarizing the content.",
            "Featured image should be 1200x630 pixels for social sharing.",
            "Target keyword density of 1-2% for primary keyword.",
            "Include call-to-action at the end of each piece.",
            "Publish consistently: 2-3 posts per week for best engagement.",
        ],
        "queries": [
            "Write about AI",
            "Create marketing copy",
            "Social media post",
            "Newsletter draft",
        ],
    },
}


def run_domain_experiment(domain_name: str, domain_data: dict, embedder_name: str = "sbert"):
    """Run experiments for a single domain."""
    from prompt_amplifier import PromptForge
    from prompt_amplifier.embedders import SentenceTransformerEmbedder

    results = {
        "domain": domain_name,
        "embedder": embedder_name,
        "num_documents": len(domain_data["documents"]),
        "num_queries": len(domain_data["queries"]),
        "query_results": [],
    }

    # Create forge with SentenceTransformers (works offline)
    forge = PromptForge(embedder=SentenceTransformerEmbedder())
    forge.add_texts(domain_data["documents"], source=domain_name)

    for query in domain_data["queries"]:
        # Search
        search_start = time.time()
        search_results = forge.search(query, top_k=3)
        search_time = (time.time() - search_start) * 1000

        # Calculate retrieval metrics
        top_scores = [r.score for r in search_results.results[:3]]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0

        results["query_results"].append({
            "query": query,
            "search_time_ms": round(search_time, 2),
            "num_results": len(search_results.results),
            "top_score": round(top_scores[0], 3) if top_scores else 0,
            "avg_score": round(avg_score, 3),
        })

    # Aggregate metrics
    results["avg_search_time_ms"] = round(
        sum(r["search_time_ms"] for r in results["query_results"]) / len(results["query_results"]),
        2,
    )
    results["avg_top_score"] = round(
        sum(r["top_score"] for r in results["query_results"]) / len(results["query_results"]),
        3,
    )

    return results


def run_ablation_chunk_size():
    """Ablation study: Effect of chunk size on retrieval quality."""
    from prompt_amplifier import PromptForge
    from prompt_amplifier.embedders import SentenceTransformerEmbedder
    from prompt_amplifier.chunkers import RecursiveChunker

    # Long document for chunking tests
    long_doc = """
    The quarterly business review covers multiple areas of performance.
    Sales revenue increased by 15% compared to the previous quarter.
    Customer acquisition cost decreased from $150 to $120 per customer.
    Net Promoter Score improved from 45 to 52 points.
    Employee satisfaction remained stable at 78%.
    Product development delivered 3 major features on schedule.
    Marketing campaigns achieved 120% of lead generation targets.
    Customer support resolved 95% of tickets within SLA.
    The engineering team reduced system downtime by 40%.
    International expansion added 2 new markets in APAC region.
    Partnership revenue grew to represent 25% of total revenue.
    The mobile app achieved 4.5 star rating with 50K downloads.
    """

    chunk_sizes = [100, 200, 500, 1000]
    results = []

    for size in chunk_sizes:
        chunker = RecursiveChunker(chunk_size=size, chunk_overlap=size // 5)
        forge = PromptForge(
            embedder=SentenceTransformerEmbedder(),
            chunker=chunker,
        )

        # Add the document
        start = time.time()
        forge.add_texts([long_doc])
        embed_time = (time.time() - start) * 1000

        # Search
        search_start = time.time()
        search_results = forge.search("sales performance", top_k=3)
        search_time = (time.time() - search_start) * 1000

        results.append({
            "chunk_size": size,
            "num_chunks": forge.chunk_count,
            "embed_time_ms": round(embed_time, 1),
            "search_time_ms": round(search_time, 1),
            "top_score": round(search_results.results[0].score, 3) if search_results.results else 0,
        })

    return results


def run_ablation_top_k():
    """Ablation study: Effect of top-k on retrieval quality."""
    from prompt_amplifier import PromptForge
    from prompt_amplifier.embedders import SentenceTransformerEmbedder

    forge = PromptForge(embedder=SentenceTransformerEmbedder())
    forge.add_texts(DOMAIN_DATA["sales"]["documents"])

    top_k_values = [1, 3, 5, 10]
    results = []

    for k in top_k_values:
        query = "deal health status"

        start = time.time()
        search_results = forge.search(query, top_k=k)
        search_time = (time.time() - start) * 1000

        scores = [r.score for r in search_results.results]

        results.append({
            "top_k": k,
            "actual_results": len(search_results.results),
            "search_time_ms": round(search_time, 2),
            "avg_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "min_score": round(min(scores), 3) if scores else 0,
            "max_score": round(max(scores), 3) if scores else 0,
        })

    return results


def run_caching_benchmark():
    """Benchmark caching performance."""
    from prompt_amplifier import PromptForge, MemoryCache, DiskCache, CacheConfig
    from prompt_amplifier.embedders import SentenceTransformerEmbedder
    import tempfile

    results = {"memory_cache": {}, "disk_cache": {}, "no_cache": {}}

    # Prepare forge with memory cache
    forge_mem = PromptForge(
        embedder=SentenceTransformerEmbedder(),
        cache=MemoryCache(CacheConfig(ttl_seconds=3600)),
    )
    forge_mem.add_texts(DOMAIN_DATA["sales"]["documents"])

    # Memory cache test
    queries = DOMAIN_DATA["sales"]["queries"]

    # First pass - all misses
    times_first = []
    for q in queries:
        start = time.time()
        forge_mem.search(q)
        times_first.append((time.time() - start) * 1000)

    # Second pass - all hits
    times_second = []
    for q in queries:
        start = time.time()
        forge_mem.search(q)
        times_second.append((time.time() - start) * 1000)

    stats = forge_mem.get_cache_stats()

    results["memory_cache"] = {
        "avg_time_first_pass_ms": round(sum(times_first) / len(times_first), 2),
        "avg_time_second_pass_ms": round(sum(times_second) / len(times_second), 2),
        "speedup": round(
            (sum(times_first) / len(times_first)) / (sum(times_second) / len(times_second)), 1
        )
        if sum(times_second) > 0
        else 0,
        "hit_rate": round(stats["hit_rate"], 2),
        "total_hits": stats["hits"],
        "total_misses": stats["misses"],
    }

    # Disk cache test
    with tempfile.TemporaryDirectory() as tmpdir:
        forge_disk = PromptForge(
            embedder=SentenceTransformerEmbedder(),
            cache=DiskCache(CacheConfig(cache_dir=tmpdir)),
        )
        forge_disk.add_texts(DOMAIN_DATA["sales"]["documents"])

        # First pass
        disk_times_first = []
        for q in queries:
            start = time.time()
            forge_disk.search(q)
            disk_times_first.append((time.time() - start) * 1000)

        # Second pass
        disk_times_second = []
        for q in queries:
            start = time.time()
            forge_disk.search(q)
            disk_times_second.append((time.time() - start) * 1000)

        disk_stats = forge_disk.get_cache_stats()

        results["disk_cache"] = {
            "avg_time_first_pass_ms": round(sum(disk_times_first) / len(disk_times_first), 2),
            "avg_time_second_pass_ms": round(sum(disk_times_second) / len(disk_times_second), 2),
            "speedup": round(
                (sum(disk_times_first) / len(disk_times_first))
                / (sum(disk_times_second) / len(disk_times_second)),
                1,
            )
            if sum(disk_times_second) > 0
            else 0,
            "hit_rate": round(disk_stats["hit_rate"], 2),
        }

    # No cache baseline
    forge_no_cache = PromptForge(
        embedder=SentenceTransformerEmbedder(),
        enable_cache=False,
    )
    forge_no_cache.add_texts(DOMAIN_DATA["sales"]["documents"])

    no_cache_times = []
    for _ in range(2):  # Two passes
        for q in queries:
            start = time.time()
            forge_no_cache.search(q)
            no_cache_times.append((time.time() - start) * 1000)

    results["no_cache"] = {
        "avg_time_ms": round(sum(no_cache_times) / len(no_cache_times), 2),
    }

    return results


def main():
    """Run all experiments."""
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARKS FOR PRIME RESEARCH PAPER")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "domain_experiments": {},
        "ablation_chunk_size": [],
        "ablation_top_k": [],
        "caching_benchmark": {},
    }

    # 1. Multi-domain experiments
    print("\n[1/4] Running multi-domain experiments...")
    for domain_name, domain_data in DOMAIN_DATA.items():
        print(f"  - {domain_name}...", end=" ", flush=True)
        try:
            domain_result = run_domain_experiment(domain_name, domain_data)
            results["domain_experiments"][domain_name] = domain_result
            print(f"✓ (avg score: {domain_result['avg_top_score']:.3f})")
        except Exception as e:
            print(f"✗ ({e})")
            results["domain_experiments"][domain_name] = {"error": str(e)}

    # 2. Ablation: Chunk size
    print("\n[2/4] Running chunk size ablation...")
    try:
        results["ablation_chunk_size"] = run_ablation_chunk_size()
        print("  ✓ Complete")
    except Exception as e:
        print(f"  ✗ ({e})")

    # 3. Ablation: Top-k
    print("\n[3/4] Running top-k ablation...")
    try:
        results["ablation_top_k"] = run_ablation_top_k()
        print("  ✓ Complete")
    except Exception as e:
        print(f"  ✗ ({e})")

    # 4. Caching benchmark
    print("\n[4/4] Running caching benchmark...")
    try:
        results["caching_benchmark"] = run_caching_benchmark()
        print("  ✓ Complete")
    except Exception as e:
        print(f"  ✗ ({e})")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n📊 Domain Results:")
    for domain, data in results["domain_experiments"].items():
        if "error" not in data:
            print(f"  {domain}: avg_score={data['avg_top_score']:.3f}, "
                  f"avg_time={data['avg_search_time_ms']:.1f}ms")

    print("\n📐 Chunk Size Ablation:")
    for item in results["ablation_chunk_size"]:
        print(f"  size={item['chunk_size']}: chunks={item['num_chunks']}, "
              f"score={item['top_score']:.3f}")

    print("\n🎯 Top-K Ablation:")
    for item in results["ablation_top_k"]:
        print(f"  k={item['top_k']}: avg_score={item['avg_score']:.3f}, "
              f"time={item['search_time_ms']:.1f}ms")

    print("\n⚡ Caching Performance:")
    cache = results["caching_benchmark"]
    if "memory_cache" in cache:
        print(f"  Memory Cache: {cache['memory_cache']['speedup']}x speedup, "
              f"{cache['memory_cache']['hit_rate']:.0%} hit rate")
    if "disk_cache" in cache:
        print(f"  Disk Cache: {cache['disk_cache']['speedup']}x speedup, "
              f"{cache['disk_cache']['hit_rate']:.0%} hit rate")

    return results


if __name__ == "__main__":
    main()

