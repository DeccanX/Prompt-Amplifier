"""LLM generators for prompt expansion."""

from __future__ import annotations

from prompt_amplifier.generators.base import BaseGenerator, GenerationResult

__all__ = [
    "BaseGenerator",
    "GenerationResult",
]

# Optional generators - OpenAI
try:
    from prompt_amplifier.generators.openai import OpenAIGenerator

    __all__.append("OpenAIGenerator")
except ImportError:
    pass

# Optional generators - Anthropic
try:
    from prompt_amplifier.generators.openai import AnthropicGenerator

    __all__.append("AnthropicGenerator")
except ImportError:
    pass

# Optional generators - Google
try:
    from prompt_amplifier.generators.openai import GoogleGenerator

    __all__.append("GoogleGenerator")
except ImportError:
    pass

# Optional generators - Ollama (local)
try:
    from prompt_amplifier.generators.ollama import OllamaGenerator

    __all__.append("OllamaGenerator")
except ImportError:
    pass

# Optional generators - Mistral
try:
    from prompt_amplifier.generators.ollama import MistralGenerator

    __all__.append("MistralGenerator")
except ImportError:
    pass

# Optional generators - Together AI
try:
    from prompt_amplifier.generators.ollama import TogetherGenerator

    __all__.append("TogetherGenerator")
except ImportError:
    pass
