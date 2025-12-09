# Tutorial: Sales Intelligence System

Build a sales intelligence system that transforms vague queries into detailed reports.

## Overview

In this tutorial, you'll create a system that:

- Loads sales documentation (playbooks, product info, deal stages)
- Understands sales-specific queries
- Generates comprehensive, structured prompts for sales reports

## Prerequisites

```bash
pip install prompt-amplifier[all]
```

## Step 1: Set Up the Knowledge Base

Create sample sales documents:

```python
# sales_knowledge.py

SALES_DOCS = [
    # Deal Stages
    """
    Deal Stages:
    1. Discovery - Understanding customer needs and pain points
    2. Technical Validation - POC and technical assessment
    3. Business Validation - ROI analysis and stakeholder alignment
    4. Negotiation - Contract terms and pricing
    5. Closed Won/Lost - Final outcome
    """,
    
    # Health Indicators
    """
    POC Health Check Guidelines:
    - Healthy: All milestones on track, positive customer engagement, >80 Winscore
    - Warning: Minor delays, some blockers, 60-80 Winscore
    - Critical: Major blockers, low engagement, <60 Winscore
    """,
    
    # Key Metrics
    """
    Key Sales Metrics:
    - Winscore (0-100): Overall deal confidence score
    - Feature Fit (%): How well product matches requirements
    - Customer Engagement: Frequency and quality of interactions
    - Days in Stage: Time spent in current deal stage
    - Stakeholder Mapping: Decision makers identified and engaged
    """,
    
    # Success Plan
    """
    Success Plan Components:
    - Deal Name and Account Executive
    - POC Duration and Timeline
    - Success Criteria (measurable goals)
    - Milestones with dates
    - Blockers and Risks
    - Next Steps and Action Items
    """,
    
    # Reporting Format
    """
    Standard Report Format:
    1. Executive Summary - 2-3 sentence overview
    2. Key Metrics Table - Winscore, Feature Fit, Engagement
    3. Current Status - Stage, health, timeline
    4. Risks and Blockers - Issues requiring attention
    5. Recommended Actions - Prioritized next steps
    """
]
```

## Step 2: Initialize the System

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import SentenceTransformerEmbedder
from prompt_amplifier.vectorstores import ChromaStore
from prompt_amplifier.core.config import PromptForgeConfig, GeneratorConfig

# Configuration
config = PromptForgeConfig(
    generator=GeneratorConfig(
        provider="openai",
        model="gpt-4o-mini"
    ),
    chunk_size=400,
    chunk_overlap=50
)

# Initialize with persistent storage
forge = PromptForge(
    config=config,
    embedder=SentenceTransformerEmbedder(),
    vectorstore=ChromaStore(
        collection_name="sales_intelligence",
        persist_directory="./sales_db"
    )
)

# Load knowledge base
forge.add_texts(SALES_DOCS)
print(f"Loaded {forge.chunk_count} chunks")
```

## Step 3: Test Common Queries

```python
# Common sales queries
queries = [
    "How's the deal going?",
    "Give me a health check",
    "What are the risks?",
    "Summarize the account",
    "What should I focus on?",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    
    result = forge.expand(query)
    print(f"Expansion: {result.expansion_ratio:.1f}x")
    print(f"\nExpanded Prompt:\n{result.prompt[:800]}...")
```

## Step 4: Create a Helper Class

```python
class SalesAssistant:
    """Sales intelligence assistant powered by Prompt Amplifier."""
    
    def __init__(self, db_path: str = "./sales_db"):
        self.forge = PromptForge(
            config=PromptForgeConfig(
                generator=GeneratorConfig(provider="openai")
            ),
            embedder=SentenceTransformerEmbedder(),
            vectorstore=ChromaStore(
                collection_name="sales",
                persist_directory=db_path
            )
        )
    
    def load_knowledge(self, docs: list[str]):
        """Load sales documentation."""
        self.forge.add_texts(docs)
        return f"Loaded {self.forge.chunk_count} chunks"
    
    def get_deal_health(self, deal_name: str = None) -> str:
        """Get deal health assessment prompt."""
        query = f"Health check for {deal_name}" if deal_name else "Deal health check"
        result = self.forge.expand(query)
        return result.prompt
    
    def get_risk_analysis(self) -> str:
        """Get risk analysis prompt."""
        result = self.forge.expand("Analyze risks and blockers")
        return result.prompt
    
    def get_next_steps(self) -> str:
        """Get recommended next steps prompt."""
        result = self.forge.expand("What should I do next?")
        return result.prompt
    
    def custom_query(self, query: str) -> str:
        """Process any sales-related query."""
        result = self.forge.expand(query)
        return result.prompt

# Usage
assistant = SalesAssistant()
assistant.load_knowledge(SALES_DOCS)

print(assistant.get_deal_health("Acme Corp"))
print(assistant.get_risk_analysis())
print(assistant.get_next_steps())
```

## Step 5: Add Real Documents

Load actual sales documents:

```python
# Load from files
assistant.forge.load_documents("./sales_playbook.pdf")
assistant.forge.load_documents("./product_catalog.docx")
assistant.forge.load_documents("./deal_templates/")

# Or from CRM export
import json

with open("deals_export.json") as f:
    deals = json.load(f)

deal_texts = [
    f"Deal: {d['name']}, Stage: {d['stage']}, Winscore: {d['winscore']}"
    for d in deals
]

assistant.forge.add_texts(deal_texts)
```

## Step 6: Integrate with Your Workflow

### As a CLI Tool

```python
# sales_cli.py
import sys
from sales_assistant import SalesAssistant

assistant = SalesAssistant()
assistant.load_knowledge(SALES_DOCS)

if len(sys.argv) > 1:
    query = " ".join(sys.argv[1:])
    print(assistant.custom_query(query))
else:
    print("Usage: python sales_cli.py <your query>")
```

```bash
python sales_cli.py "How is the Acme Corp deal going?"
```

### As an API Endpoint

```python
# sales_api.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
assistant = SalesAssistant()
assistant.load_knowledge(SALES_DOCS)

class Query(BaseModel):
    text: str

@app.post("/expand")
def expand_prompt(query: Query):
    result = assistant.forge.expand(query.text)
    return {
        "expanded_prompt": result.prompt,
        "expansion_ratio": result.expansion_ratio
    }

@app.get("/health/{deal_name}")
def deal_health(deal_name: str):
    return {"prompt": assistant.get_deal_health(deal_name)}
```

## Complete Example

```python
# complete_sales_system.py

from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import SentenceTransformerEmbedder
from prompt_amplifier.vectorstores import ChromaStore
from prompt_amplifier.core.config import PromptForgeConfig, GeneratorConfig

# Sales knowledge base
SALES_KNOWLEDGE = [
    "Deal Stages: Discovery, Technical Validation, Business Validation, Negotiation, Closed",
    "POC Health: Healthy (>80 Winscore), Warning (60-80), Critical (<60)",
    "Key Metrics: Winscore, Feature Fit %, Engagement Score, Days in Stage",
    "Success Plan: Deal Name, POC Duration, Success Criteria, Milestones, Blockers",
    "Report Format: Executive Summary, Metrics Table, Status, Risks, Actions",
]

def main():
    # Initialize
    forge = PromptForge(
        config=PromptForgeConfig(
            generator=GeneratorConfig(provider="openai", model="gpt-4o-mini")
        ),
        embedder=SentenceTransformerEmbedder(),
        vectorstore=ChromaStore(
            collection_name="sales",
            persist_directory="./db"
        )
    )
    
    # Load knowledge
    forge.add_texts(SALES_KNOWLEDGE)
    
    # Interactive mode
    print("Sales Intelligence System")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("Your query: ").strip()
        if query.lower() == 'quit':
            break
        
        result = forge.expand(query)
        print(f"\n{'='*60}")
        print(f"Expansion: {result.expansion_ratio:.1f}x")
        print(f"{'='*60}")
        print(result.prompt)
        print()

if __name__ == "__main__":
    main()
```

## Summary

You've learned how to:

1. ✅ Set up a sales-specific knowledge base
2. ✅ Configure Prompt Amplifier for sales queries
3. ✅ Create a reusable SalesAssistant class
4. ✅ Integrate with CLI and API workflows
5. ✅ Load real documents and CRM data

## Next Steps

- [Research Assistant Tutorial](research-assistant.md)
- [Customer Support Tutorial](customer-support.md)

