# Generators

Generators use LLMs to expand prompts based on retrieved context.

## Available Generators

| Generator | Provider | Default Model |
|-----------|----------|---------------|
| `OpenAIGenerator` | OpenAI | gpt-4o-mini |
| `AnthropicGenerator` | Anthropic | claude-3-haiku |
| `GoogleGenerator` | Google | gemini-2.0-flash |

## OpenAI

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.core.config import PromptForgeConfig, GeneratorConfig

config = PromptForgeConfig(
    generator=GeneratorConfig(provider="openai", model="gpt-4o")
)

forge = PromptForge(config=config)
```

## Anthropic

```python
config = PromptForgeConfig(
    generator=GeneratorConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
)

forge = PromptForge(config=config)
```

## Google Gemini

```python
config = PromptForgeConfig(
    generator=GeneratorConfig(provider="google", model="gemini-2.0-flash")
)

forge = PromptForge(config=config)
```

## Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
```

