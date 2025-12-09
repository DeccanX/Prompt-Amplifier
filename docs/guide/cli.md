# Command-Line Interface

Prompt Amplifier includes a powerful CLI for quick prompt expansion, document search, and evaluation without writing code.

## Installation

The CLI is included when you install Prompt Amplifier:

```bash
pip install prompt-amplifier
```

Verify installation:

```bash
prompt-amplifier version
```

---

## Commands

### `expand` - Expand a Prompt

Transform a short prompt into a detailed one:

```bash
# Basic expansion
prompt-amplifier expand "How's the deal going?"

# With documents for context
prompt-amplifier expand "Summarize performance" --docs ./sales_data/

# With inline text context
prompt-amplifier expand "Analyze trends" --texts "Q4 revenue was $5M" "Growth was 23%"

# Choose embedder
prompt-amplifier expand "Check status" --docs ./docs/ --embedder sentence-transformers

# Choose generator
prompt-amplifier expand "Explain metrics" --docs ./docs/ --generator anthropic

# Specify model
prompt-amplifier expand "Summarize" --docs ./docs/ --generator openai --model gpt-4-turbo

# JSON output
prompt-amplifier expand "Status check" --docs ./docs/ --json
```

#### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--docs` | `-d` | Path to documents directory |
| `--texts` | `-t` | Inline text strings for context |
| `--embedder` | `-e` | Embedder: tfidf, bm25, sentence-transformers, openai, google |
| `--generator` | `-g` | Generator: openai, anthropic, google, ollama |
| `--model` | `-m` | Specific model name |
| `--top-k` | `-k` | Number of context chunks (default: 5) |
| `--json` | `-j` | Output as JSON |

---

### `search` - Search Documents

Search your documents without LLM expansion:

```bash
# Basic search
prompt-amplifier search "customer satisfaction" --docs ./docs/

# Limit results
prompt-amplifier search "revenue growth" --docs ./docs/ --top-k 3

# JSON output
prompt-amplifier search "Q4 performance" --docs ./docs/ --json
```

#### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--docs` | `-d` | Path to documents directory |
| `--texts` | `-t` | Inline text strings to search |
| `--top-k` | `-k` | Number of results (default: 5) |
| `--json` | `-j` | Output as JSON |

---

### `compare-embedders` - Benchmark Embedders

Compare different embedding strategies:

```bash
# Compare with default texts
prompt-amplifier compare-embedders

# With your documents
prompt-amplifier compare-embedders --docs ./docs/

# With specific queries
prompt-amplifier compare-embedders --texts "ML basics" "NLP intro" --queries "What is AI?"
```

#### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--docs` | `-d` | Path to documents directory |
| `--texts` | `-t` | Inline text strings |
| `--queries` | `-q` | Test queries for comparison |

#### Example Output

```
Embedder Comparison
============================================================

TF-IDF:
  Dimension: 4
  Embedding time: 2.3ms
  Query time: 0.1ms

Sentence Transformers:
  Dimension: 384
  Embedding time: 125.4ms
  Query time: 12.3ms
```

---

### `evaluate` - Run Evaluation Suite

Run comprehensive evaluations:

```bash
# Basic evaluation
prompt-amplifier evaluate --docs ./docs/

# With specific prompts
prompt-amplifier evaluate --docs ./docs/ --prompts "Status check" "Metrics summary"
```

#### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--docs` | `-d` | Path to documents directory |
| `--prompts` | `-p` | Test prompts to evaluate |

#### Example Output

```
======================================================================
EVALUATION REPORT
======================================================================

📝 Test: Test 1
   Prompt: How's the deal going?...
   ✅ Success
   ⏱️  Time: 1523ms
   📊 Expansion: 8.2x
   🎯 Quality: 0.75

📝 Test: Test 2
   Prompt: Check project status...
   ✅ Success
   ⏱️  Time: 1456ms
   📊 Expansion: 7.5x
   🎯 Quality: 0.82

======================================================================
Summary: 2/2 tests passed
Average Quality: 0.79
Average Expansion: 7.9x
```

---

### `version` - Show Version

Display the installed version:

```bash
prompt-amplifier version
# prompt-amplifier 0.2.0
```

---

## Environment Variables

Set API keys via environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="AIza..."

# Or use a .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## Examples

### Quick Demo

```bash
# Create sample data
mkdir -p demo_docs
echo "Q4 revenue reached $5.2M with 23% growth." > demo_docs/sales.txt
echo "Customer satisfaction score is 4.5/5." > demo_docs/metrics.txt

# Expand a prompt
prompt-amplifier expand "How are we doing?" --docs ./demo_docs/
```

### Batch Processing

```bash
# Process multiple prompts
for prompt in "Status check" "Metrics summary" "Forecast"; do
  echo "=== $prompt ==="
  prompt-amplifier expand "$prompt" --docs ./docs/ --json
done
```

### Save Results

```bash
# Save to file
prompt-amplifier expand "Analyze performance" --docs ./docs/ --json > result.json

# View structure
cat result.json | jq .expansion_ratio
```

### Integration with Scripts

```python
import subprocess
import json

result = subprocess.run(
    ["prompt-amplifier", "expand", "Status check", "--docs", "./docs/", "--json"],
    capture_output=True,
    text=True,
)
data = json.loads(result.stdout)
print(f"Expansion ratio: {data['expansion_ratio']:.1f}x")
```

---

## Troubleshooting

### "No documents loaded"

```bash
# Ensure docs directory has files
ls -la ./docs/

# Check supported formats: .txt, .pdf, .docx, .csv, .json
```

### "API key missing"

```bash
# Set the required API key
export OPENAI_API_KEY="sk-..."

# Or use local embedder/generator
prompt-amplifier expand "Test" --docs ./docs/ --embedder tfidf --generator ollama
```

### "Module not found"

```bash
# Install optional dependencies
pip install prompt-amplifier[all]

# Or specific ones
pip install prompt-amplifier[embeddings-openai,generators-openai]
```

