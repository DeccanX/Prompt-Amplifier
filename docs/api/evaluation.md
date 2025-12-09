# Evaluation API Reference

## PromptMetrics

```python
@dataclass
class PromptMetrics:
    """Metrics for evaluating expanded prompt quality."""
    
    expansion_ratio: float = 0.0
    structure_score: float = 0.0
    specificity_score: float = 0.0
    completeness_score: float = 0.0
    readability_score: float = 0.0
    overall_score: float = 0.0
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `expansion_ratio` | `float` | Ratio of expanded to original length |
| `structure_score` | `float` | Score for structural elements (0-1) |
| `specificity_score` | `float` | Score for specific instructions (0-1) |
| `completeness_score` | `float` | Score for section coverage (0-1) |
| `readability_score` | `float` | Score for readability (0-1) |
| `overall_score` | `float` | Weighted overall score (0-1) |
| `details` | `dict` | Detailed breakdown of scores |

---

## RetrievalMetrics

```python
@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""
    
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    average_score: float = 0.0
    relevance_scores: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `precision_at_k` | `float` | Precision at k documents |
| `recall_at_k` | `float` | Recall at k documents |
| `mrr` | `float` | Mean Reciprocal Rank |
| `ndcg` | `float` | Normalized Discounted Cumulative Gain |
| `average_score` | `float` | Average similarity score |
| `relevance_scores` | `list[float]` | Individual relevance scores |

---

## Functions

### calculate_expansion_quality

```python
def calculate_expansion_quality(
    original_prompt: str,
    expanded_prompt: str,
    weights: Optional[dict] = None,
) -> PromptMetrics:
    """
    Calculate quality metrics for an expanded prompt.
    
    Args:
        original_prompt: The original short prompt.
        expanded_prompt: The expanded prompt.
        weights: Optional weights for overall score. Default:
            {"structure": 0.25, "specificity": 0.25, 
             "completeness": 0.25, "readability": 0.25}
    
    Returns:
        PromptMetrics with all calculated scores.
    """
```

#### Example

```python
from prompt_amplifier.evaluation import calculate_expansion_quality

metrics = calculate_expansion_quality(
    original_prompt="Summarize data",
    expanded_prompt="**GOAL:** Generate summary...",
    weights={"structure": 0.4, "specificity": 0.3, 
             "completeness": 0.2, "readability": 0.1}
)
```

---

### calculate_retrieval_metrics

```python
def calculate_retrieval_metrics(
    retrieved_scores: list[float],
    relevant_indices: Optional[list[int]] = None,
    k: int = 5,
) -> RetrievalMetrics:
    """
    Calculate retrieval quality metrics.
    
    Args:
        retrieved_scores: Similarity scores of retrieved documents.
        relevant_indices: Indices of known relevant documents.
        k: Number of retrieved documents to consider.
    
    Returns:
        RetrievalMetrics with calculated scores.
    """
```

#### Example

```python
from prompt_amplifier.evaluation import calculate_retrieval_metrics

metrics = calculate_retrieval_metrics(
    retrieved_scores=[0.9, 0.8, 0.6, 0.4, 0.3],
    relevant_indices=[0, 1, 4],
    k=5
)
```

---

### calculate_diversity_score

```python
def calculate_diversity_score(embeddings: list[list[float]]) -> float:
    """
    Calculate diversity score for embeddings.
    
    Higher score means more diverse (less similar) results.
    
    Args:
        embeddings: List of embedding vectors.
    
    Returns:
        Diversity score (0-1).
    """
```

#### Example

```python
from prompt_amplifier.evaluation import calculate_diversity_score

score = calculate_diversity_score([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
])
```

---

### calculate_coherence_score

```python
def calculate_coherence_score(
    prompt: str,
    context_chunks: list[str],
) -> float:
    """
    Calculate coherence between prompt and context.
    
    Args:
        prompt: The expanded prompt.
        context_chunks: Context strings used for expansion.
    
    Returns:
        Coherence score (0-1).
    """
```

#### Example

```python
from prompt_amplifier.evaluation import calculate_coherence_score

score = calculate_coherence_score(
    prompt="Analyze Q4 sales in North America...",
    context_chunks=["Q4 sales reached $1.2M in North America"]
)
```

---

### compare_embedders

```python
def compare_embedders(
    texts: list[str],
    queries: list[str],
    embedders: list[Any],
    embedder_names: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Compare multiple embedders on the same data.
    
    Args:
        texts: Documents to embed.
        queries: Queries to test retrieval.
        embedders: List of embedder instances.
        embedder_names: Optional names for embedders.
    
    Returns:
        Dictionary with comparison results:
        {
            "EmbedderName": {
                "embedding_time_ms": float,
                "query_time_ms": float,
                "dimension": int,
                "avg_query_scores": list[float],
                "error": str  # if failed
            }
        }
    """
```

#### Example

```python
from prompt_amplifier.evaluation import compare_embedders
from prompt_amplifier.embedders import TFIDFEmbedder, SentenceTransformerEmbedder

results = compare_embedders(
    texts=["doc1", "doc2", "doc3"],
    queries=["query1", "query2"],
    embedders=[TFIDFEmbedder(), SentenceTransformerEmbedder()],
    embedder_names=["TF-IDF", "Sentence Transformers"]
)
```

---

### benchmark_generators

```python
def benchmark_generators(
    prompt: str,
    context: str,
    generators: list[Any],
    generator_names: Optional[list[str]] = None,
    num_runs: int = 1,
) -> dict[str, dict]:
    """
    Benchmark multiple generators on the same prompt.
    
    Args:
        prompt: The prompt to expand.
        context: Context for expansion.
        generators: List of generator instances.
        generator_names: Optional names for generators.
        num_runs: Number of runs to average.
    
    Returns:
        Dictionary with benchmark results:
        {
            "GeneratorName": {
                "avg_time_ms": float,
                "avg_expansion_ratio": float,
                "avg_quality_score": float,
                "outputs": list[str],
                "error": str  # if failed
            }
        }
    """
```

#### Example

```python
from prompt_amplifier.evaluation import benchmark_generators
from prompt_amplifier.generators import OpenAIGenerator, AnthropicGenerator

results = benchmark_generators(
    prompt="Summarize performance",
    context="Q4 revenue was $5.2M...",
    generators=[OpenAIGenerator(), AnthropicGenerator()],
    generator_names=["GPT-4", "Claude"],
    num_runs=3
)
```

---

## EvaluationSuite

```python
class EvaluationSuite:
    """Comprehensive evaluation suite for Prompt Amplifier."""
    
    def __init__(self):
        self.test_cases: list[dict] = []
    
    def add_test_case(
        self,
        name: str,
        prompt: str,
        relevant_docs: Optional[list[str]] = None,
        expected_keywords: Optional[list[str]] = None,
    ) -> None:
        """Add a test case to the suite."""
    
    def run(self, forge: Any) -> list[dict]:
        """Run all test cases against a PromptForge instance."""
    
    def print_report(self, results: list[dict]) -> None:
        """Print a formatted report of results."""
```

### Methods

#### add_test_case

```python
def add_test_case(
    self,
    name: str,
    prompt: str,
    relevant_docs: Optional[list[str]] = None,
    expected_keywords: Optional[list[str]] = None,
) -> None:
    """
    Add a test case.
    
    Args:
        name: Name of the test case.
        prompt: The prompt to test.
        relevant_docs: Optional list of relevant document contents.
        expected_keywords: Optional keywords to check in output.
    """
```

#### run

```python
def run(self, forge: Any) -> list[dict]:
    """
    Run all test cases.
    
    Args:
        forge: PromptForge instance to test.
    
    Returns:
        List of result dictionaries:
        [
            {
                "name": str,
                "prompt": str,
                "success": bool,
                "time_ms": float,
                "expanded_prompt": str,
                "expansion_ratio": float,
                "quality_metrics": dict,
                "keyword_coverage": float,  # if keywords provided
                "error": str  # if failed
            }
        ]
    """
```

#### print_report

```python
def print_report(self, results: list[dict]) -> None:
    """
    Print formatted report.
    
    Args:
        results: Results from run().
    """
```

### Complete Example

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.evaluation import EvaluationSuite

# Setup
forge = PromptForge()
forge.add_texts(["POC Health tracking...", "Winscore metrics..."])

# Create suite
suite = EvaluationSuite()
suite.add_test_case(
    name="Deal Status",
    prompt="How's the deal?",
    expected_keywords=["POC", "health"]
)
suite.add_test_case(
    name="Metrics",
    prompt="What metrics to track?",
    expected_keywords=["Winscore"]
)

# Run and report
results = suite.run(forge)
suite.print_report(results)
```

