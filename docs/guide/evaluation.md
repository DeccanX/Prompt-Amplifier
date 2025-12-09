# Evaluation Metrics

Prompt Amplifier includes a comprehensive evaluation module for measuring prompt quality, retrieval accuracy, and comparing different configurations.

## Prompt Quality Metrics

Measure the quality of expanded prompts with `calculate_expansion_quality`:

```python
from prompt_amplifier.evaluation import calculate_expansion_quality

original = "Summarize the sales data"
expanded = """**GOAL:** Generate a comprehensive summary of sales data.

**SECTIONS:**
1. Executive Overview
2. Key Performance Indicators
3. Trend Analysis
4. Recommendations

**INSTRUCTIONS:**
- Include quarterly comparisons
- Highlight top-performing products
- Format numbers with proper currency symbols
"""

metrics = calculate_expansion_quality(original, expanded)

print(f"Expansion Ratio: {metrics.expansion_ratio:.1f}x")
print(f"Structure Score: {metrics.structure_score:.2f}")
print(f"Specificity Score: {metrics.specificity_score:.2f}")
print(f"Completeness Score: {metrics.completeness_score:.2f}")
print(f"Readability Score: {metrics.readability_score:.2f}")
print(f"Overall Score: {metrics.overall_score:.2f}")
```

### What Each Metric Measures

| Metric | Description | Range |
|--------|-------------|-------|
| **Expansion Ratio** | Length increase from original | 1x - N x |
| **Structure Score** | Headers, bullets, numbered lists | 0.0 - 1.0 |
| **Specificity Score** | Action verbs, constraints, examples | 0.0 - 1.0 |
| **Completeness Score** | Goal, sections, instructions present | 0.0 - 1.0 |
| **Readability Score** | Sentence length appropriateness | 0.0 - 1.0 |
| **Overall Score** | Weighted combination | 0.0 - 1.0 |

---

## Retrieval Metrics

Evaluate retrieval quality with standard IR metrics:

```python
from prompt_amplifier.evaluation import calculate_retrieval_metrics

# Similarity scores from retrieval
scores = [0.92, 0.85, 0.71, 0.58, 0.42]

# Ground truth: indices 0, 1, 4 are truly relevant
relevant = [0, 1, 4]

metrics = calculate_retrieval_metrics(
    retrieved_scores=scores,
    relevant_indices=relevant,
    k=5
)

print(f"Precision@5: {metrics.precision_at_k:.2f}")
print(f"Recall@5: {metrics.recall_at_k:.2f}")
print(f"MRR: {metrics.mrr:.2f}")
print(f"NDCG: {metrics.ndcg:.2f}")
print(f"Average Score: {metrics.average_score:.2f}")
```

### Retrieval Metrics Explained

| Metric | Description |
|--------|-------------|
| **Precision@k** | Fraction of retrieved docs that are relevant |
| **Recall@k** | Fraction of relevant docs that were retrieved |
| **MRR** | Mean Reciprocal Rank - position of first relevant |
| **NDCG** | Normalized Discounted Cumulative Gain |
| **Average Score** | Mean similarity score of retrieved docs |

---

## Diversity Score

Measure how diverse your retrieved results are:

```python
from prompt_amplifier.evaluation import calculate_diversity_score

# Embeddings of retrieved documents
embeddings = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
]

diversity = calculate_diversity_score(embeddings)
print(f"Diversity: {diversity:.2f}")
# Higher = more diverse results
```

---

## Coherence Score

Measure how well the expanded prompt uses the context:

```python
from prompt_amplifier.evaluation import calculate_coherence_score

prompt = "Analyze quarterly sales for North America region..."
context_chunks = [
    "Q1 sales in North America reached $1.2M",
    "North American market shows 15% growth",
]

coherence = calculate_coherence_score(prompt, context_chunks)
print(f"Coherence: {coherence:.2f}")
# Higher = prompt better incorporates context
```

---

## Compare Embedders

Benchmark different embedders on your data:

```python
from prompt_amplifier.evaluation import compare_embedders
from prompt_amplifier.embedders import (
    TFIDFEmbedder,
    SentenceTransformerEmbedder,
)

texts = [
    "Machine learning fundamentals",
    "Deep neural networks",
    "Natural language processing",
    "Computer vision applications",
]

queries = [
    "How does NLP work?",
    "Explain deep learning",
]

results = compare_embedders(
    texts=texts,
    queries=queries,
    embedders=[TFIDFEmbedder(), SentenceTransformerEmbedder()],
    embedder_names=["TF-IDF", "Sentence Transformers"],
)

for name, data in results.items():
    print(f"\n{name}:")
    print(f"  Dimension: {data['dimension']}")
    print(f"  Embedding time: {data['embedding_time_ms']:.1f}ms")
    print(f"  Query time: {data['query_time_ms']:.1f}ms")
    print(f"  Avg scores: {data['avg_query_scores']}")
```

### Example Output

```
TF-IDF:
  Dimension: 4
  Embedding time: 2.3ms
  Query time: 0.1ms
  Avg scores: [0.15, 0.22]

Sentence Transformers:
  Dimension: 384
  Embedding time: 125.4ms
  Query time: 12.3ms
  Avg scores: [0.78, 0.85]
```

---

## Benchmark Generators

Compare different LLMs for prompt expansion:

```python
from prompt_amplifier.evaluation import benchmark_generators
from prompt_amplifier.generators import (
    OpenAIGenerator,
    AnthropicGenerator,
    GoogleGenerator,
)

results = benchmark_generators(
    prompt="Summarize Q4 performance",
    context="Q4 revenue was $5.2M with 23% growth...",
    generators=[
        OpenAIGenerator(),
        AnthropicGenerator(),
        GoogleGenerator(),
    ],
    generator_names=["GPT-4", "Claude", "Gemini"],
    num_runs=3,  # Average over multiple runs
)

for name, data in results.items():
    print(f"\n{name}:")
    print(f"  Avg time: {data['avg_time_ms']:.0f}ms")
    print(f"  Avg expansion: {data['avg_expansion_ratio']:.1f}x")
    print(f"  Avg quality: {data['avg_quality_score']:.2f}")
```

---

## Evaluation Suite

Run comprehensive evaluations with the `EvaluationSuite`:

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.evaluation import EvaluationSuite

# Setup PromptForge with your data
forge = PromptForge()
forge.add_texts([
    "POC Health: Healthy means all milestones on track.",
    "Winscore ranges from 0-100, higher is better.",
    "Feature fit percentage indicates product match.",
])

# Create evaluation suite
suite = EvaluationSuite()

# Add test cases
suite.add_test_case(
    name="Deal Status",
    prompt="How's the deal going?",
    expected_keywords=["POC", "health", "milestone"],
)
suite.add_test_case(
    name="Metrics Check",
    prompt="What metrics should I track?",
    expected_keywords=["Winscore", "feature fit"],
)
suite.add_test_case(
    name="Product Fit",
    prompt="Is our product a good fit?",
    expected_keywords=["feature", "fit", "percentage"],
)

# Run all tests
results = suite.run(forge)

# Print formatted report
suite.print_report(results)
```

### Sample Report Output

```
======================================================================
EVALUATION REPORT
======================================================================

📝 Test: Deal Status
   Prompt: How's the deal going?...
   ✅ Success
   ⏱️  Time: 1523ms
   📊 Expansion: 8.2x
   🎯 Quality: 0.75
   🔑 Keywords: 100%

📝 Test: Metrics Check
   Prompt: What metrics should I track?...
   ✅ Success
   ⏱️  Time: 1456ms
   📊 Expansion: 7.5x
   🎯 Quality: 0.82
   🔑 Keywords: 100%

📝 Test: Product Fit
   Prompt: Is our product a good fit?...
   ✅ Success
   ⏱️  Time: 1389ms
   📊 Expansion: 6.9x
   🎯 Quality: 0.71
   🔑 Keywords: 67%

======================================================================
Summary: 3/3 tests passed
Average Quality: 0.76
Average Expansion: 7.5x
```

---

## CLI Evaluation

Run evaluations from command line:

```bash
# Compare embedders
prompt-amplifier compare-embedders --docs ./docs/

# Run evaluation suite
prompt-amplifier evaluate --docs ./docs/ --prompts "How's the deal?" "Check metrics"
```

---

## Best Practices

### 1. Create Representative Test Cases

```python
suite.add_test_case(
    name="Edge case: Very short",
    prompt="Hi",
)
suite.add_test_case(
    name="Edge case: Technical query",
    prompt="What's the MTTR for critical incidents?",
)
```

### 2. Track Metrics Over Time

```python
import json
from datetime import datetime

results = suite.run(forge)
metrics = {
    "timestamp": datetime.now().isoformat(),
    "version": "0.2.0",
    "avg_quality": sum(r["quality_metrics"]["overall_score"] 
                       for r in results if r["success"]) / len(results),
}

with open("metrics_history.jsonl", "a") as f:
    f.write(json.dumps(metrics) + "\n")
```

### 3. A/B Test Configurations

```python
configs = [
    {"embedder": TFIDFEmbedder(), "name": "TF-IDF"},
    {"embedder": SentenceTransformerEmbedder(), "name": "ST"},
]

for config in configs:
    forge = PromptForge(embedder=config["embedder"])
    forge.add_texts(texts)
    results = suite.run(forge)
    print(f"{config['name']}: {results[0]['quality_metrics']['overall_score']:.2f}")
```

