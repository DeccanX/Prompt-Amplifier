# Additional Generators

Prompt Amplifier v0.2.0 adds support for local LLMs and additional cloud providers for prompt expansion.

## Ollama (Local LLMs)

Run LLMs locally with no API costs or data leaving your machine.

```python
from prompt_amplifier.generators import OllamaGenerator

# Default model (llama2)
generator = OllamaGenerator()

# Specify model
generator = OllamaGenerator(
    model="llama3.1:8b",
    base_url="http://localhost:11434",  # Default Ollama URL
)

# Generate
expanded = generator.generate(
    prompt="Summarize the data",
    context="Q4 revenue was $5.2M...",
)
```

### Popular Ollama Models

| Model | Size | Best For |
|-------|------|----------|
| `llama3.1:8b` | 8B | General purpose |
| `llama3.1:70b` | 70B | High quality |
| `mistral:7b` | 7B | Fast, good quality |
| `codellama:13b` | 13B | Code generation |
| `phi3:mini` | 3.8B | Resource constrained |

### Setup

1. Install Ollama: https://ollama.ai
2. Pull a model:
```bash
ollama pull llama3.1:8b
```
3. Use in code:
```python
generator = OllamaGenerator(model="llama3.1:8b")
```

### Installation

```bash
pip install prompt-amplifier[generators-ollama]
```

---

## Mistral AI Generator

High-quality European LLM with excellent performance.

```python
from prompt_amplifier.generators import MistralGenerator

generator = MistralGenerator(
    api_key="your-mistral-api-key",  # or MISTRAL_API_KEY env var
    model="mistral-large-latest",
)

expanded = generator.generate(
    prompt="Explain the metrics",
    context="Winscore measures deal probability...",
)
```

### Available Models

| Model | Context | Best For |
|-------|---------|----------|
| `mistral-tiny` | 32K | Fast, cheap |
| `mistral-small-latest` | 32K | Balanced |
| `mistral-medium-latest` | 32K | Good quality |
| `mistral-large-latest` | 32K | Best quality |

### Features

- GDPR compliant
- Strong multilingual
- Competitive pricing
- Function calling support

### Installation

```bash
pip install prompt-amplifier[generators-mistral]
```

---

## Together AI Generator

Access multiple open-source models through one API.

```python
from prompt_amplifier.generators import TogetherGenerator

generator = TogetherGenerator(
    api_key="your-together-api-key",  # or TOGETHER_API_KEY env var
    model="meta-llama/Llama-3-70b-chat-hf",
)

expanded = generator.generate(
    prompt="Analyze the pipeline",
    context="Current pipeline value is $2.5M...",
)
```

### Available Models

| Model | Provider | Size |
|-------|----------|------|
| `meta-llama/Llama-3-70b-chat-hf` | Meta | 70B |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | Mistral | 8x7B |
| `Qwen/Qwen2-72B-Instruct` | Alibaba | 72B |
| `google/gemma-2-27b-it` | Google | 27B |

### Features

- Access to 50+ models
- Pay-per-token pricing
- Model comparison easy
- Fast inference

### Installation

```bash
pip install prompt-amplifier[generators-together]
```

---

## Comparison Table

| Provider | Local | Cost | Quality | Speed | Context |
|----------|-------|------|---------|-------|---------|
| OpenAI GPT-4 | ❌ | $$$$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 128K |
| OpenAI GPT-3.5 | ❌ | $ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 16K |
| Anthropic Claude 3 | ❌ | $$$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 200K |
| Google Gemini | ❌ | $$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 1M |
| Mistral Large | ❌ | $$ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 32K |
| Together Llama 3 | ❌ | $ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8K |
| Ollama (local) | ✅ | Free | ⭐⭐⭐⭐ | ⭐⭐⭐ | Varies |

---

## Choosing a Generator

### For Privacy/Offline
```python
# Runs completely locally
from prompt_amplifier.generators import OllamaGenerator
generator = OllamaGenerator(model="llama3.1:8b")
```

### For Best Quality
```python
# GPT-4 or Claude 3 Opus
from prompt_amplifier.generators import OpenAIGenerator
generator = OpenAIGenerator(model="gpt-4-turbo")
```

### For Budget Production
```python
# Good quality, low cost
from prompt_amplifier.generators import TogetherGenerator
generator = TogetherGenerator(model="meta-llama/Llama-3-70b-chat-hf")
```

### For GDPR Compliance
```python
# EU-based
from prompt_amplifier.generators import MistralGenerator
generator = MistralGenerator()
```

### For Long Context
```python
# 1M token context
from prompt_amplifier.generators import GoogleGenerator
generator = GoogleGenerator(model="gemini-1.5-pro")
```

---

## Using Multiple Generators

Compare outputs from different generators:

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.generators import (
    OpenAIGenerator,
    OllamaGenerator,
)
from prompt_amplifier.evaluation import benchmark_generators

# Create generators
generators = [
    OpenAIGenerator(model="gpt-4-turbo"),
    OllamaGenerator(model="llama3.1:8b"),
]

# Benchmark
results = benchmark_generators(
    prompt="How's the deal going?",
    context="POC is healthy, all milestones on track.",
    generators=generators,
    generator_names=["GPT-4 Turbo", "Llama 3.1 8B"],
)

for name, data in results.items():
    print(f"\n{name}:")
    print(f"  Time: {data['avg_time_ms']:.0f}ms")
    print(f"  Quality: {data['avg_quality_score']:.2f}")
```

---

## Dynamic Generator Selection

Choose generator based on requirements:

```python
import os
from prompt_amplifier import PromptForge

def get_generator(preference: str = "quality"):
    if preference == "quality" and os.getenv("OPENAI_API_KEY"):
        from prompt_amplifier.generators import OpenAIGenerator
        return OpenAIGenerator(model="gpt-4-turbo")
    
    elif preference == "budget" and os.getenv("TOGETHER_API_KEY"):
        from prompt_amplifier.generators import TogetherGenerator
        return TogetherGenerator()
    
    elif preference == "local":
        from prompt_amplifier.generators import OllamaGenerator
        return OllamaGenerator()
    
    else:
        # Fallback to Gemini (often has free tier)
        from prompt_amplifier.generators import GoogleGenerator
        return GoogleGenerator()

# Usage
forge = PromptForge(generator=get_generator("local"))
```

