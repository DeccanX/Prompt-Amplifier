# Tutorial: Research Assistant

Build a research assistant that helps analyze papers and generate detailed research queries.

## Overview

Create a system that:

- Loads research papers and academic documents
- Understands research-specific terminology
- Generates comprehensive research prompts

## Prerequisites

```bash
pip install prompt-amplifier[all]
```

## Step 1: Set Up Research Knowledge Base

```python
RESEARCH_KNOWLEDGE = [
    # Research methodology
    """
    Research Paper Structure:
    - Abstract: Brief summary of the work
    - Introduction: Problem statement and motivation
    - Related Work: Prior research in the field
    - Methodology: Approach and techniques used
    - Experiments: Setup, datasets, and results
    - Discussion: Analysis and implications
    - Conclusion: Summary and future work
    """,
    
    # Analysis frameworks
    """
    Paper Analysis Framework:
    1. Problem Definition: What problem does it solve?
    2. Novel Contribution: What's new compared to prior work?
    3. Methodology: How does it work?
    4. Evaluation: How is it validated?
    5. Limitations: What are the weaknesses?
    6. Future Directions: What comes next?
    """,
    
    # Common research tasks
    """
    Research Tasks:
    - Literature Review: Survey existing work on a topic
    - Comparative Analysis: Compare multiple approaches
    - Gap Analysis: Identify unexplored areas
    - Replication Study: Reproduce published results
    - Extension Study: Build upon existing work
    """,
    
    # Quality criteria
    """
    Paper Quality Criteria:
    - Novelty: Original contribution to the field
    - Significance: Impact on the research community
    - Soundness: Technical correctness
    - Reproducibility: Can results be replicated?
    - Clarity: Well-written and organized
    """
]
```

## Step 2: Create the Research Assistant

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import SentenceTransformerEmbedder
from prompt_amplifier.vectorstores import ChromaStore
from prompt_amplifier.core.config import PromptForgeConfig, GeneratorConfig

class ResearchAssistant:
    """AI research assistant powered by Prompt Amplifier."""
    
    def __init__(self, db_path: str = "./research_db"):
        config = PromptForgeConfig(
            generator=GeneratorConfig(
                provider="anthropic",  # Claude is great for research
                model="claude-3-5-sonnet-20241022"
            ),
            chunk_size=600,
            chunk_overlap=100
        )
        
        self.forge = PromptForge(
            config=config,
            embedder=SentenceTransformerEmbedder(
                model="all-mpnet-base-v2"  # Better for academic text
            ),
            vectorstore=ChromaStore(
                collection_name="research",
                persist_directory=db_path
            )
        )
    
    def load_papers(self, path: str):
        """Load research papers from directory."""
        self.forge.load_documents(path)
        return f"Loaded {self.forge.chunk_count} chunks"
    
    def add_knowledge(self, texts: list[str]):
        """Add research knowledge."""
        self.forge.add_texts(texts)
    
    def analyze_paper(self, paper_topic: str) -> str:
        """Generate paper analysis prompt."""
        return self.forge.expand(
            f"Analyze the research paper about {paper_topic}"
        ).prompt
    
    def literature_review(self, topic: str) -> str:
        """Generate literature review prompt."""
        return self.forge.expand(
            f"Literature review on {topic}"
        ).prompt
    
    def compare_approaches(self, approaches: list[str]) -> str:
        """Generate comparison prompt."""
        return self.forge.expand(
            f"Compare these approaches: {', '.join(approaches)}"
        ).prompt
    
    def identify_gaps(self, field: str) -> str:
        """Generate gap analysis prompt."""
        return self.forge.expand(
            f"Identify research gaps in {field}"
        ).prompt
    
    def summarize(self, topic: str) -> str:
        """Generate summary prompt."""
        return self.forge.expand(
            f"Summarize {topic}"
        ).prompt

# Usage
assistant = ResearchAssistant()
assistant.add_knowledge(RESEARCH_KNOWLEDGE)

# Load actual papers
assistant.load_papers("./papers/")
```

## Step 3: Example Queries

```python
# Analyze a specific paper
prompt = assistant.analyze_paper("transformer architectures")
print(prompt)

# Generate literature review
prompt = assistant.literature_review("prompt engineering techniques")
print(prompt)

# Compare approaches
prompt = assistant.compare_approaches([
    "BERT", "GPT", "T5", "LLaMA"
])
print(prompt)

# Find research gaps
prompt = assistant.identify_gaps("retrieval-augmented generation")
print(prompt)
```

## Step 4: Add Your Papers

```python
# Load PDFs
assistant.load_papers("./papers/attention_is_all_you_need.pdf")
assistant.load_papers("./papers/bert.pdf")
assistant.load_papers("./papers/gpt3.pdf")

# Or add summaries directly
paper_summaries = [
    """
    Paper: Attention Is All You Need (2017)
    Authors: Vaswani et al.
    Contribution: Introduced the Transformer architecture
    Key Innovation: Self-attention mechanism replacing RNNs
    Impact: Foundation for BERT, GPT, and modern LLMs
    """,
    """
    Paper: BERT (2018)
    Authors: Devlin et al.
    Contribution: Bidirectional pre-training for NLP
    Key Innovation: Masked language modeling
    Impact: State-of-the-art on many NLP benchmarks
    """,
]

assistant.add_knowledge(paper_summaries)
```

## Step 5: Interactive Research Session

```python
def research_session():
    assistant = ResearchAssistant()
    assistant.add_knowledge(RESEARCH_KNOWLEDGE)
    
    print("Research Assistant Ready")
    print("Commands: analyze, review, compare, gaps, quit\n")
    
    while True:
        cmd = input("Command: ").strip().lower()
        
        if cmd == "quit":
            break
        elif cmd == "analyze":
            topic = input("Paper topic: ")
            print(assistant.analyze_paper(topic))
        elif cmd == "review":
            topic = input("Review topic: ")
            print(assistant.literature_review(topic))
        elif cmd == "compare":
            approaches = input("Approaches (comma-separated): ").split(",")
            print(assistant.compare_approaches(approaches))
        elif cmd == "gaps":
            field = input("Research field: ")
            print(assistant.identify_gaps(field))
        else:
            # Free-form query
            print(assistant.forge.expand(cmd).prompt)
        
        print()

if __name__ == "__main__":
    research_session()
```

## Output Example

**Query:** "Analyze transformer architectures"

**Expanded Prompt:**
```markdown
**GOAL:** Conduct a comprehensive analysis of transformer architectures

**REQUIRED SECTIONS:**

1. **Overview**
   - Definition of transformer architecture
   - Historical context and development

2. **Technical Analysis**
   - Self-attention mechanism
   - Multi-head attention
   - Positional encoding
   - Feed-forward networks

3. **Comparative Analysis**
   | Model | Parameters | Training Data | Key Innovation |
   |-------|------------|---------------|----------------|
   | BERT  | ...        | ...           | ...            |
   | GPT   | ...        | ...           | ...            |

4. **Strengths and Limitations**
   - Advantages over previous approaches
   - Known limitations and challenges

5. **Applications**
   - NLP tasks
   - Computer vision
   - Multi-modal learning

6. **Future Directions**
   - Efficiency improvements
   - Scaling laws
   - Novel architectures

**INSTRUCTIONS:**
- Use technical terminology appropriate for ML researchers
- Include citations where relevant
- Focus on architectural innovations
...
```

## Summary

You've built a research assistant that:

- ✅ Loads and indexes research papers
- ✅ Understands academic terminology
- ✅ Generates structured analysis prompts
- ✅ Supports literature reviews and comparisons
- ✅ Identifies research gaps

## Next Steps

- [Customer Support Tutorial](customer-support.md)
- [Sales Intelligence Tutorial](sales-intelligence.md)

