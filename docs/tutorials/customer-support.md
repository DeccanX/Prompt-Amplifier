# Tutorial: Customer Support System

Build a customer support system that generates detailed support responses.

## Overview

Create a system that:

- Loads product documentation and FAQs
- Understands customer queries
- Generates comprehensive support prompts

## Prerequisites

```bash
pip install prompt-amplifier[all]
```

## Step 1: Set Up Support Knowledge Base

```python
SUPPORT_KNOWLEDGE = [
    # Product information
    """
    Product: CloudSync Pro
    Type: Cloud storage and sync solution
    Plans: Free (5GB), Pro ($9.99/mo, 100GB), Business ($29.99/mo, 1TB)
    Features: File sync, sharing, collaboration, version history
    Platforms: Windows, Mac, iOS, Android, Web
    """,
    
    # Common issues
    """
    Common Issues and Solutions:
    1. Sync not working
       - Check internet connection
       - Verify account status
       - Restart the app
       - Check file size limits
    
    2. Login problems
       - Reset password via email
       - Clear browser cache
       - Check for service status
    
    3. Storage full
       - Upgrade plan
       - Delete old files
       - Empty trash
    """,
    
    # Support tiers
    """
    Support Response Guidelines:
    - Tier 1: Basic troubleshooting, FAQ answers
    - Tier 2: Technical issues, account problems
    - Tier 3: Billing, refunds, enterprise support
    
    Response Format:
    1. Acknowledge the issue
    2. Provide solution steps
    3. Offer alternatives if needed
    4. Include relevant links
    5. Set expectations for resolution
    """,
    
    # Tone guidelines
    """
    Communication Guidelines:
    - Be empathetic and professional
    - Use clear, simple language
    - Avoid technical jargon when possible
    - Personalize responses with customer name
    - End with a helpful question or offer
    """
]
```

## Step 2: Create Support Assistant

```python
from prompt_amplifier import PromptForge
from prompt_amplifier.embedders import SentenceTransformerEmbedder
from prompt_amplifier.vectorstores import ChromaStore
from prompt_amplifier.core.config import PromptForgeConfig, GeneratorConfig

class SupportAssistant:
    """Customer support assistant powered by Prompt Amplifier."""
    
    def __init__(self, db_path: str = "./support_db"):
        config = PromptForgeConfig(
            generator=GeneratorConfig(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.7  # Slightly creative for empathetic responses
            ),
            chunk_size=400,
            chunk_overlap=50
        )
        
        self.forge = PromptForge(
            config=config,
            embedder=SentenceTransformerEmbedder(),
            vectorstore=ChromaStore(
                collection_name="support",
                persist_directory=db_path
            )
        )
    
    def load_docs(self, path: str):
        """Load support documentation."""
        self.forge.load_documents(path)
    
    def add_faqs(self, faqs: list[str]):
        """Add FAQ entries."""
        self.forge.add_texts(faqs)
    
    def handle_query(self, query: str, customer_name: str = "Customer") -> str:
        """Generate support response prompt."""
        full_query = f"Customer {customer_name} asks: {query}"
        return self.forge.expand(full_query).prompt
    
    def troubleshoot(self, issue: str) -> str:
        """Generate troubleshooting prompt."""
        return self.forge.expand(f"Troubleshoot: {issue}").prompt
    
    def explain_feature(self, feature: str) -> str:
        """Generate feature explanation prompt."""
        return self.forge.expand(f"Explain feature: {feature}").prompt
    
    def billing_query(self, query: str) -> str:
        """Handle billing-related queries."""
        return self.forge.expand(f"Billing question: {query}").prompt

# Initialize
assistant = SupportAssistant()
assistant.add_faqs(SUPPORT_KNOWLEDGE)
```

## Step 3: Example Support Scenarios

```python
# Scenario 1: Technical issue
prompt = assistant.handle_query(
    "My files aren't syncing",
    customer_name="John"
)
print("SYNC ISSUE PROMPT:")
print(prompt)
print()

# Scenario 2: Feature question
prompt = assistant.explain_feature("version history")
print("FEATURE EXPLANATION PROMPT:")
print(prompt)
print()

# Scenario 3: Billing
prompt = assistant.billing_query("How do I upgrade my plan?")
print("BILLING PROMPT:")
print(prompt)
print()

# Scenario 4: Troubleshooting
prompt = assistant.troubleshoot("App crashes on startup")
print("TROUBLESHOOTING PROMPT:")
print(prompt)
```

## Step 4: Add Real Documentation

```python
# Load from various sources
assistant.load_docs("./docs/user_manual.pdf")
assistant.load_docs("./docs/faq.md")
assistant.load_docs("./docs/troubleshooting_guide.docx")

# Add common Q&A pairs
qa_pairs = [
    "Q: How do I reset my password? A: Go to Settings > Account > Reset Password",
    "Q: What's the file size limit? A: 5GB for free, 50GB for Pro, unlimited for Business",
    "Q: Can I share files with non-users? A: Yes, create a public link in the share menu",
]
assistant.add_faqs(qa_pairs)
```

## Step 5: Support Ticket System

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class Ticket:
    id: str
    customer_name: str
    email: str
    subject: str
    description: str
    priority: Priority
    created_at: datetime

class SupportTicketSystem:
    """Complete support ticket system."""
    
    def __init__(self):
        self.assistant = SupportAssistant()
        self.assistant.add_faqs(SUPPORT_KNOWLEDGE)
        self.tickets = {}
    
    def create_ticket(self, customer_name: str, email: str,
                      subject: str, description: str) -> Ticket:
        """Create a new support ticket."""
        ticket_id = f"TKT-{len(self.tickets) + 1:04d}"
        
        # Auto-classify priority
        priority = self._classify_priority(description)
        
        ticket = Ticket(
            id=ticket_id,
            customer_name=customer_name,
            email=email,
            subject=subject,
            description=description,
            priority=priority,
            created_at=datetime.now()
        )
        
        self.tickets[ticket_id] = ticket
        return ticket
    
    def _classify_priority(self, description: str) -> Priority:
        """Classify ticket priority based on content."""
        urgent_keywords = ["urgent", "emergency", "down", "broken", "asap"]
        high_keywords = ["not working", "error", "failed", "crash"]
        
        desc_lower = description.lower()
        
        if any(kw in desc_lower for kw in urgent_keywords):
            return Priority.URGENT
        elif any(kw in desc_lower for kw in high_keywords):
            return Priority.HIGH
        else:
            return Priority.MEDIUM
    
    def generate_response(self, ticket_id: str) -> str:
        """Generate response prompt for a ticket."""
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            return "Ticket not found"
        
        return self.assistant.handle_query(
            ticket.description,
            customer_name=ticket.customer_name
        )
    
    def get_open_tickets(self) -> list[Ticket]:
        """Get all open tickets sorted by priority."""
        priority_order = {
            Priority.URGENT: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3
        }
        return sorted(
            self.tickets.values(),
            key=lambda t: priority_order[t.priority]
        )

# Usage
system = SupportTicketSystem()

# Create ticket
ticket = system.create_ticket(
    customer_name="Alice Smith",
    email="alice@example.com",
    subject="Can't sync files",
    description="My files haven't synced for 2 days. I've tried restarting."
)

print(f"Created: {ticket.id} (Priority: {ticket.priority.value})")

# Generate response
response_prompt = system.generate_response(ticket.id)
print(f"\nSuggested Response Prompt:\n{response_prompt}")
```

## Step 6: Integration Example

```python
# support_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Support API")
system = SupportTicketSystem()

class TicketCreate(BaseModel):
    customer_name: str
    email: str
    subject: str
    description: str

class QueryRequest(BaseModel):
    query: str
    customer_name: str = "Customer"

@app.post("/tickets")
def create_ticket(ticket: TicketCreate):
    t = system.create_ticket(
        customer_name=ticket.customer_name,
        email=ticket.email,
        subject=ticket.subject,
        description=ticket.description
    )
    return {"ticket_id": t.id, "priority": t.priority.value}

@app.get("/tickets/{ticket_id}/response")
def get_response(ticket_id: str):
    prompt = system.generate_response(ticket_id)
    if prompt == "Ticket not found":
        raise HTTPException(status_code=404, detail="Ticket not found")
    return {"response_prompt": prompt}

@app.post("/query")
def quick_query(request: QueryRequest):
    prompt = system.assistant.handle_query(
        request.query,
        request.customer_name
    )
    return {"prompt": prompt}
```

## Summary

You've built a customer support system that:

- ✅ Loads product documentation
- ✅ Handles various query types
- ✅ Auto-classifies ticket priority
- ✅ Generates empathetic response prompts
- ✅ Integrates with ticketing systems

## Next Steps

- [Sales Intelligence Tutorial](sales-intelligence.md)
- [Research Assistant Tutorial](research-assistant.md)

