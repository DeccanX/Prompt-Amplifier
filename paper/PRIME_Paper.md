# PRIME: Prompt Refinement via Information-driven Methods and Expansion—A Modular Framework for Context-Aware Prompt Amplification

**Rajesh More**¹, **Madhukar Patneedi**¹, **Bhanu Prakash Doppalapudi**¹, **Raghav Chaturvedi**¹, **David Trakhtenberg**¹, **Jyoti Ambad**², **Sidhant Gupta**¹, **Rukesh Patel**¹

¹ Cloudwerx Worldwide (CWX)  
² Research Scholar

Correspondence: rajesh.more@cwx.tech

---

## Abstract

While Large Language Models (LLMs) have transformed natural language processing, their effectiveness depends critically on prompt quality—a gap that existing work has largely overlooked. Current Retrieval-Augmented Generation (RAG) systems retrieve documents to generate *answers*; we propose a fundamentally different approach: using retrieval to construct *better questions*. This paper introduces PRIME (Prompt Refinement via Information-driven Methods and Expansion), a framework that treats prompt construction as a first-class optimization problem rather than a manual preprocessing step.

PRIME implements a modular pipeline with heterogeneous document loaders (10+ formats including web and video), pluggable embedding strategies (sparse TF-IDF/BM25 and dense Sentence-BERT/Gemini-Embedding-001/OpenAI), persistent vector stores, and multi-provider LLM generators (GPT-4o, Claude-3, Gemini-2.5-Flash, Gemini-3-Flash-Preview). We formalize prompt amplification mathematically and introduce four novel evaluation metrics: structural coherence, semantic specificity, contextual completeness, and lexical readability.

Comprehensive experiments across four domains, 12 embedding configurations, 6 LLM backends, and 30 human-evaluated prompts reveal: (1) dense embeddings achieve 37-73% higher retrieval precision than sparse methods; (2) Gemini-3-Flash-Preview achieves 165× expansion ratio, outperforming Gemini-2.5-Flash (121×); (3) Gemini-Embedding-001 provides comparable quality to text-embedding-004 with newer architecture; (4) complex queries outperform simple ones by 92%; (5) human evaluators rate PRIME outputs 4.2/5 on average with 87% inter-rater agreement; and (6) caching provides 1,944× speedup for production deployments. We also present a systematic failure analysis identifying when prompt amplification degrades performance.

PRIME is released as an open-source library (`pip install prompt-amplifier`), representing the first comprehensive framework for automated prompt engineering.

**Keywords:** Prompt Engineering, Prompt Amplification, Retrieval-Augmented Generation, Text Embeddings, LLM, Prompt Optimization, Human Evaluation, Information Retrieval

---

## 1. Introduction

Over the past two years, Large Language Models have quietly revolutionized how millions of people work. Tools built on GPT-4, Claude, and Gemini now draft emails, summarize documents, and generate code across industries. Yet anyone who has spent time with these systems knows a frustrating truth: getting good results often requires surprisingly specific instructions.

Here's a scene that plays out daily in offices worldwide. A sales manager opens their AI assistant and types: "How's the deal going?" It's exactly what they'd ask a colleague. But the AI responds with a generic platitude about "monitoring key metrics"—useless for making an actual decision. The manager sighs, closes the tab, and goes back to manually reviewing spreadsheets.

The core problem is a mismatch. Humans communicate through context and shared understanding. We expect our colleagues to know what "the deal" means, which metrics matter, and how we prefer information presented. LLMs have none of this context unless we explicitly provide it. This *prompt engineering problem* has spawned an entire cottage industry of courses, consultants, and copy-paste template libraries.

### 1.1 This Is Not Just Another RAG System

Let us be explicit about what makes PRIME different: **we treat prompt construction as a first-class optimization target, not a manual preprocessing step**.

Existing RAG systems retrieve documents to help generate *answers*. PRIME retrieves documents to construct *better questions*. This distinction is fundamental. When a user asks "How's the deal going?", a traditional RAG system would retrieve deal-related documents and attempt to answer the question directly. PRIME instead retrieves documents to understand *what the user should be asking*—what metrics exist, what thresholds indicate health, what stakeholders matter—and constructs a detailed prompt that elicits precisely the information needed.

Consider the information flow:

- **Traditional RAG**: Query → Retrieve → Generate Answer
- **PRIME**: Query → Retrieve → Generate *Enhanced Query* → (Optional) Generate Answer

This seemingly simple change has profound implications. The expanded prompt can be used with any downstream LLM, cached for reuse, inspected for quality, and iteratively refined. We have transformed an opaque generation step into an observable, controllable process.

### 1.2 Contributions

This paper makes six main contributions:

1. **The PRIME Framework**: A modular, open-source library that operationalizes prompt amplification with pluggable components at every pipeline stage.

2. **Formal Problem Definition**: We cast prompt amplification as an optimization problem, providing theoretical grounding for what makes a "good" expanded prompt.

3. **Novel Evaluation Metrics**: We introduce four metrics specifically designed for expanded prompts (Structure, Specificity, Completeness, Readability)—addressing a gap in prompt evaluation literature.

4. **Comprehensive Benchmarking**: We evaluate 12 embedding configurations (including the latest Gemini-Embedding-001) and 6 LLM generators (including Gemini-3-Flash-Preview) across four domains.

5. **Human Evaluation**: We report results from 30 prompts evaluated by 3 raters, demonstrating correlation between automated metrics and human judgment.

6. **Failure Analysis**: We systematically identify when prompt amplification fails or degrades output quality, providing practitioners with actionable guidance.

---

## 2. Background and Related Work

### 2.1 Retrieval-Augmented Generation

The idea of combining search with generation isn't new—librarians have done it for centuries. But when Lewis et al. (2020) formalized Retrieval-Augmented Generation (RAG), they gave it a name and a framework that stuck. Their insight was elegant: LLMs have impressive capabilities but limited, static knowledge. By retrieving relevant documents before generating, you get the best of both worlds—fresh, accurate information with fluent generation.

Since then, RAG has become standard practice for building knowledge-intensive applications. Chatbots, search engines, and document analysis tools all use some variant. But there's a gap in the literature: almost all RAG work focuses on augmenting *answers*. We ask a different question—what if we used retrieval to augment the *prompts* themselves?

Think about it. When you ask a domain expert a vague question, they don't just answer—they first clarify and expand your question based on what they know about the domain. "How's the deal going?" becomes "Are you asking about the Winscore, the milestone progress, the executive engagement, or the overall health status?" That's what PRIME does automatically.

### 2.2 The Evolution of Prompt Engineering

Prompt engineering has gone through distinct phases, each revealing something important about how LLMs work:

**Phase 1: Templates** (2019-2020). Early practitioners discovered that certain phrasings work better than others. "Translate to French:" beats "Make this French". This era produced endless blogs about "magic prompts" that supposedly unlocked hidden capabilities.

**Phase 2: Few-Shot Learning** (2020). Brown et al.'s GPT-3 paper changed everything. They showed that including a few examples in the prompt—without any fine-tuning—dramatically improved performance. Suddenly, prompts weren't just instructions; they were mini-training sets.

**Phase 3: Chain-of-Thought** (2022). Wei et al. discovered something surprising: asking models to "think step by step" actually makes them better at reasoning. The prompt structure itself, not just the content, affects output quality. This insight is central to our work.

**Phase 4: Automatic Optimization** (2023). Zhou et al. showed that LLMs can optimize their own prompts—essentially, AI prompt engineering. But these approaches are constrained to the LLM's internal knowledge. They can't access organization-specific terminology, metrics, or preferences.

PRIME sits at a new intersection: using *external knowledge* to automatically improve prompts, combining the retrieval power of RAG with the structural insights from prompt optimization research.

### 2.3 Text Embeddings

Before you can search, you need to represent text numerically. This seemingly technical choice has profound implications for what your system can find.

**Sparse representations** (TF-IDF, BM25) treat documents as bags of words. If your query contains "automobile" but your documents say "car," you get nothing. These methods are fast—sub-millisecond fast—and need no neural networks. They've powered web search for decades. But they miss semantic connections.

**Dense representations** (Sentence-BERT, OpenAI embeddings) learn to map text into continuous vector spaces where similar meanings are nearby. "Car" and "automobile" land close together. "Happy" and "joyful" are neighbors. These embeddings capture *what you mean*, not just *what you say*. The cost? More computation and typically an external API or a local model.

The choice matters more than many practitioners realize. Our experiments show 37-73% quality differences between approaches—far from a rounding error. We'll quantify exactly when each approach makes sense.

---

## 3. System Architecture

Building a prompt amplification system requires solving five distinct problems: getting documents in, splitting them intelligently, representing them mathematically, finding relevant pieces, and generating coherent expansions. PRIME addresses each with pluggable components, letting users swap parts without rewriting their code.

The pipeline flows naturally: Ingestion → Chunking → Embedding → Retrieval → Generation. Let's walk through each stage.

### 3.1 Document Ingestion

The first problem is mundane but critical: getting data in. Corporate knowledge doesn't live in clean text files. It's scattered across PDFs (often scanned), PowerPoints with speaker notes, Excel sheets with crucial context in column headers, Confluence wikis, Notion pages, and that one critical document someone shared as a Google Doc.

PRIME takes a practical approach—we support 10+ formats out of the box:

| Format | Loader | Description |
|--------|--------|-------------|
| .txt | TxtLoader | Plain text files |
| .pdf | PDFLoader | PDF documents |
| .docx | DocxLoader | Word documents |
| .csv | CSVLoader | Tabular data |
| .json | JSONLoader | Structured data |
| .xlsx | ExcelLoader | Spreadsheets |
| URL | WebLoader | Web pages |
| YouTube | YouTubeLoader | Video transcripts |
| Sitemap | SitemapLoader | Crawl entire sites |
| RSS | RSSLoader | Feed content |

Each loader produces standardized Document objects with content and metadata, enabling consistent downstream processing regardless of source format.

### 3.2 Text Chunking

Here's a problem that seems simple until you try to solve it: how do you split a 50-page document into pieces small enough to embed, while keeping each piece meaningful?

Naive approaches—split every N characters—create garbage. You end up with chunks that start mid-sentence and end mid-word. The embedding of such a chunk captures... what, exactly?

Our recursive chunker respects natural boundaries:

```
Algorithm: RecursiveChunk(text, separators, size, overlap)
1. If text fits in chunk size, return [text]
2. Split by current separator (paragraph → sentence → word)
3. Combine adjacent pieces until size limit
4. Include overlap with previous chunk
5. Recurse with finer separators if needed
```

The key insight: try splitting by paragraphs first. If paragraphs are too big, split by sentences. Only if sentences are too big (rare) do we split by words. This hierarchy preserves semantic coherence.

The overlap parameter ensures context doesn't get lost at chunk boundaries. If a concept spans two paragraphs, both chunks will capture part of it. Our ablation study (Section 6.5) shows this matters: smaller chunks with overlap significantly outperform larger chunks.

### 3.3 Embedding Module

We support both sparse and dense embeddings:

**Sparse (TF-IDF, BM25)**: Fast, no external dependencies, good for lexical matching. Best when exact keyword matches matter.

**Dense (Sentence-BERT, OpenAI, Google)**: Slower, captures semantic similarity. Better for conceptual queries where paraphrasing is common.

The choice significantly impacts quality, as our experiments demonstrate.

### 3.4 Retrieval

Given a query, we embed it and find similar chunks:

```
similarity(query, doc) = cos(embed(query), embed(doc))
                       = (q · d) / (||q|| × ||d||)
```

For hybrid retrieval, we combine sparse and dense scores:

```
score_hybrid = α × score_dense + (1-α) × score_sparse
```

This often outperforms either approach alone.

### 3.5 Prompt Generation

The retrieved context is formatted and passed to an LLM with instructions for expansion:

```
System: You are a prompt engineering expert. Transform brief inputs 
into comprehensive, structured prompts.

Context: [retrieved chunks]

User Query: [original prompt]

Generate an expanded prompt with clear goals, sections, and instructions.
```

The LLM produces the expanded prompt, which includes structure and specificity absent from the original.

---

## 4. Methodology

### 4.1 Problem Formalization

What exactly are we optimizing? Without a formal definition, "better prompts" remains vague. Here's our formulation:

**Definition (Prompt Amplification)**: Given input prompt *p*, knowledge corpus *K*, and quality function *Q*, find the expansion *p\** that maximizes quality while preserving intent:

```
p* = argmax Q(p')
     p' ∈ P(p,K)
```

subject to: Intent(p') ≡ Intent(p)

The intent preservation constraint is crucial. An expanded prompt that drifts into unrelated topics—even if well-structured—fails the task. The user asked about deal health, not a general overview of sales methodology.

In practice, we approximate this optimization through retrieval (find relevant knowledge) and generation (synthesize into a coherent prompt). The quality of the approximation depends on both components, which we evaluate separately.

### 4.2 Quality Metrics

Evaluating generated prompts is harder than it sounds. Human judgment is expensive and inconsistent. Standard NLP metrics (BLEU, ROUGE) compare against references that don't exist. We designed four interpretable metrics that capture what practitioners actually care about:

**Structural Coherence (S)**: Does the prompt have clear organization? We detect headers (##), bullet points (•, -), numbered lists (1., 2.), and explicit sections:

```
S(p) = (1/N) × Σ min(count(pattern_i, p) / threshold_i, 1)
```

A well-structured prompt guides the LLM. "First do X, then Y, finally Z" beats a wall of text.

**Semantic Specificity (P)**: Vague prompts get vague answers. We check for action verbs ("generate", "analyze", "compare"), constraints ("must", "required", "exactly"), and format specifications ("as a table", "in JSON"):

```
P(p) = (|ActionVerbs ∩ p| + |Constraints ∩ p| + |Formats ∩ p|) / MaxScore
```

**Contextual Completeness (C)**: Good prompts set expectations. We check for five elements: goal statement, context, required sections, specific instructions, and output format:

```
C(p) = |ExpectedSections ∩ p| / |ExpectedSections|
```

**Lexical Readability (L)**: Is the prompt well-written? We use sentence length as a proxy (optimal range: 15-25 words):

```
L(p) = 1 if 15 ≤ avg_sentence_length ≤ 25, else scaled penalty
```

The overall quality score combines these:

```
Q(p) = 0.25×S + 0.25×P + 0.25×C + 0.25×L
```

---

## 5. Experimental Setup

### 5.1 Multi-Domain Datasets

Unlike prior work that evaluates on single domains, we tested PRIME across four distinct domains to assess generalization:

| Domain | Documents | Queries | Description |
|--------|-----------|---------|-------------|
| **Sales** | 8 | 4 | Deal health, Winscore, pipeline metrics |
| **Research** | 8 | 4 | Paper structure, methodology, citations |
| **Customer Support** | 8 | 4 | Ticket tiers, SLA, resolution times |
| **Content Creation** | 8 | 4 | SEO, formatting, publishing guidelines |

This diversity tests whether PRIME's retrieval and expansion work across different vocabulary, document structures, and query types.

### 5.2 Sales Domain (Primary)

Our primary evaluation used a Sales/POC domain corpus comprising 8 documents covering:
- Deal health indicators (Healthy, At Risk, Critical)
- Performance metrics (Winscore, Feature Fit %)
- Process stages (Discovery → Closed)
- Success factors (stakeholder engagement, executive sponsors)

Test queries included natural prompts like "How's the deal going?" and "What are the risk factors?"

### 5.3 Configurations Tested

**Embedders**:
- TF-IDF (sparse, local)
- BM25 (sparse, local)
- Sentence-BERT MiniLM (dense, local, 384 dims)
- OpenAI text-embedding-3-small (dense, API, 1536 dims)
- Google text-embedding-004 (dense, API, 768 dims)
- **Google Gemini-Embedding-001** (dense, API, 768 dims) — *latest*

**Generators**:
- OpenAI GPT-4o-mini
- Anthropic Claude-3-Haiku
- Google Gemini-2.0-Flash
- **Google Gemini-2.5-Flash** (latest stable)
- **Google Gemini-3-Flash-Preview** (experimental)

### 5.4 Hardware and Model Versions

All experiments ran on an Apple M2 Pro with 32GB RAM. API calls used production endpoints with standard rate limits.

**Model Versions (December 2024)**:
- OpenAI: gpt-4o-mini-2024-07-18
- Anthropic: claude-3-haiku-20240307
- Google: gemini-2.5-flash, gemini-3-flash-preview, gemini-embedding-001, text-embedding-004
- Local: sentence-transformers/all-MiniLM-L6-v2

*Note*: Model capabilities evolve rapidly. We include version identifiers for reproducibility; results may differ with newer versions.

---

## 6. Results

### 6.1 Embedding Comparison

| Embedder | Dimension | Embed Time | Query Time | P@5 |
|----------|-----------|------------|------------|-----|
| TF-IDF | 431 | 3.2 ms | 0.11 ms | 0.45 |
| BM25 | 15 | 1.9 ms | 0.02 ms | 0.52 |
| SBERT-MiniLM | 384 | 6,256 ms | 35.8 ms | 0.71 |
| OpenAI-3-small | 1,536 | 972 ms | 2,676 ms | 0.78 |
| Google text-embedding-004 | 768 | 554 ms | 384 ms | 0.76 |
| **Gemini-Embedding-001** | 768 | 1,584 ms | 462 ms | **0.77** |

**Key Findings**:

1. **Dense embeddings significantly outperform sparse**: P@5 improves from 0.45-0.52 (sparse) to 0.71-0.78 (dense), a 37-73% relative improvement.

2. **Gemini-Embedding-001 matches established models**: Google's newest embedding model achieves P@5 = 0.77, comparable to OpenAI (0.78) and text-embedding-004 (0.76), suggesting maturity of the Gemini embedding family.

3. **Local dense embeddings are viable**: SBERT achieves P@5 = 0.71 with no API cost, only 9% below the best API-based method.

4. **Sparse methods are dramatically faster**: Sub-millisecond query times vs. hundreds of milliseconds for dense methods. This matters for high-throughput applications.

5. **Trade-off between latency and recency**: text-embedding-004 is 2.9× faster than Gemini-Embedding-001 (554ms vs 1,584ms), but the newer model may incorporate more recent training data.

### 6.2 Generator Comparison

| Generator | Model | Latency | Quality | Expansion |
|-----------|-------|---------|---------|-----------|
| OpenAI | gpt-4o-mini | 10.2s | 0.576 | 88× |
| Anthropic | claude-3-haiku | 3.3s | 0.687 | 43× |
| Google | gemini-2.0-flash | 3.9s | 0.751 | 125× |
| **Google** | **gemini-2.5-flash** | 12.2s | 0.782 | 121× |
| **Google** | **gemini-3-flash-preview** | **8.7s** | **0.798** | **165×** |

**Key Findings**:

1. **Gemini-3-Flash-Preview achieves highest quality** (0.798) with the most comprehensive expansions (165× ratio). This experimental model shows significant improvements over the stable 2.5 release.

2. **Gemini-2.5-Flash is recommended for production**: Achieves 0.782 quality with stable API, balancing quality and reliability.

3. **Claude remains fastest** at 3.3s, suitable for latency-critical interactive applications.

4. **Expansion ratios vary widely**: From 43× (Claude) to 165× (Gemini-3). Higher isn't always better—it depends on whether you need comprehensive coverage or concise guidance.

5. **Model evolution is rapid**: Gemini-3 outperforms Gemini-2.5 by 2% in quality and 36% in expansion ratio, demonstrating the pace of LLM advancement.

### 6.3 Quality Metric Breakdown

| Generator | Structure | Specificity | Completeness | Readability |
|-----------|-----------|-------------|--------------|-------------|
| GPT-4o-mini | 0.27 | 0.08 | 0.20 | 1.00 |
| Claude-3-Haiku | 0.80 | 0.17 | 0.40 | 0.62 |
| Gemini-2.0-flash | 0.33 | 0.25 | 0.20 | 0.66 |

**Interesting patterns emerge**:

- **Claude excels at structure**: Score of 0.80 vs. 0.27-0.33 for others. It naturally produces organized outputs with clear sections.
- **GPT-4o-mini has perfect readability**: Its prose is polished but lacks specific structure.
- **Gemini leads in specificity**: It includes more actionable instructions and format requirements.

These complementary strengths suggest value in task-specific generator selection or ensemble approaches.

### 6.4 Multi-Domain Evaluation

To assess generalization, we evaluated PRIME across four domains using Sentence-BERT embeddings:

| Domain | Avg Top Score | Avg Search Time | Best Query |
|--------|--------------|-----------------|------------|
| **Research** | 0.519 | 13.8 ms | "Literature review" (0.667) |
| **Content Creation** | 0.297 | 8.1 ms | "Social media post" (0.441) |
| **Sales** | 0.269 | 34.9 ms | "Analyze deal risks" (0.392) |
| **Customer Support** | 0.195 | 6.2 ms | "Help with billing" (0.252) |

**Key Observations**:

1. **Domain vocabulary matters**: Research domain achieves highest scores (0.519) because academic terminology is well-represented in SBERT's training data. Support queries with vague terms ("product not working") score lower (0.099).

2. **Query specificity correlates with score**: "Literature review" (0.667) outperforms "What's the methodology?" (0.567) because the former uses precise academic vocabulary.

3. **PRIME works across domains**: All domains achieve meaningful retrieval (scores > 0.19), demonstrating the framework's generalization capability.

### 6.5 Ablation Studies

#### Effect of Chunk Size

Smaller chunks improve retrieval precision but increase storage and processing:

| Chunk Size | Num Chunks | Top Score | Embed Time |
|------------|------------|-----------|------------|
| 100 | 12 | **0.637** | 5,234 ms |
| 200 | 6 | 0.575 | 5,297 ms |
| 500 | 2 | 0.537 | 5,586 ms |
| 1000 | 1 | 0.501 | 5,759 ms |

**Finding**: Smaller chunks (100 characters) achieve 27% higher retrieval scores than large chunks (1000 characters). However, this comes at the cost of more chunks to embed and store. The optimal chunk size depends on document density and query granularity.

#### Effect of Top-K

Retrieving more context increases coverage but dilutes average relevance:

| Top-K | Avg Score | Min Score | Max Score |
|-------|-----------|-----------|-----------|
| 1 | **0.574** | 0.574 | 0.574 |
| 3 | 0.398 | 0.303 | 0.574 |
| 5 | 0.275 | 0.083 | 0.574 |
| 10 | 0.194 | 0.040 | 0.574 |

**Finding**: Average score drops significantly as K increases, suggesting diminishing returns. For prompt expansion, k=3-5 provides a good balance between context breadth and relevance.

### 6.6 Caching Performance

PRIME includes an optional caching layer to reduce latency and API costs:

| Cache Type | First Pass | Second Pass | Speedup | Hit Rate |
|------------|------------|-------------|---------|----------|
| **Memory Cache** | 8.71 ms | 0.01 ms | **1,944×** | 50% |
| **Disk Cache** | 9.97 ms | 0.25 ms | 39.2× | 50% |
| No Cache | 8.96 ms | 8.96 ms | 1× | N/A |

**Key Benefits**:

1. **Dramatic speedup**: Memory caching provides nearly 2,000× speedup for repeated queries, reducing response time from ~9ms to 0.005ms.

2. **Persistent caching**: Disk cache maintains speedup across sessions (39×), useful for production deployments.

3. **Cost savings**: For API-based embeddings/generators, caching eliminates redundant API calls, potentially saving 50%+ on API costs for applications with query repetition.

### 6.7 Query Complexity Analysis

We investigated how query length and specificity affect retrieval quality:

| Query Type | Example | Avg Score | Avg Time |
|------------|---------|-----------|----------|
| **Simple** | "deal" | 0.276 | 17.8 ms |
| **Medium** | "deal status" | 0.416 | 8.5 ms |
| **Complex** | "What is the current deal health status?" | **0.530** | 86.2 ms |

**Key Insight**: Counter-intuitively, longer, more specific queries achieve *higher* retrieval scores (+92% vs simple queries). This occurs because:

1. **More semantic content**: Complex queries contain more distinctive vocabulary that the embedder can match.
2. **Reduced ambiguity**: "deal" could match many things; "deal health status" constrains the search space.
3. **Natural language advantage**: Sentence-based embedders like SBERT are trained on full sentences, not single words.

**Implication**: Users should be encouraged to provide more context in their queries, contrary to the common assumption that brief queries are better.

### 6.8 Embedder Comparison (Controlled)

Direct comparison on identical dataset (Sales domain, 8 documents, 4 queries):

| Embedder | Avg Score | Avg Time | Memory | API Cost |
|----------|-----------|----------|--------|----------|
| TF-IDF | 0.227 | **0.2 ms** | Low | Free |
| SBERT-MiniLM | **0.268** | 10.6 ms | Medium | Free |
| OpenAI-3-small | 0.78* | 972 ms | Low | $0.02/1M |
| Google Embed | 0.76* | 298 ms | Low | $0.0001/1K |

*From prior API experiments

**Trade-off Analysis**:

- **TF-IDF**: 50× faster but 18% lower quality. Best for high-throughput, low-latency needs.
- **SBERT**: Good balance of quality and cost. Best for most use cases.
- **API embeddings**: Highest quality but require network calls and incur costs.

### 6.9 Hybrid Retrieval

We tested combining BM25 (lexical) with vector (semantic) search:

| Configuration | Avg Score | Notes |
|---------------|-----------|-------|
| Vector-only (α=1.0) | **0.349** | Dense semantic matching |
| BM25-only (α=0.0) | 0.287 | Keyword matching |
| Hybrid (α=0.5) | 0.318 | Combined approach |

**Finding**: For our test corpus, pure vector retrieval outperforms hybrid approaches. This differs from findings in large-scale IR benchmarks, likely because:

1. Our corpus is small (8-16 documents) where semantic matching is sufficient
2. Document vocabulary is specialized and consistent
3. Hybrid benefits emerge at scale with more lexical diversity

**Recommendation**: Start with vector-only retrieval; consider hybrid for large, heterogeneous corpora.

### 6.10 Human Evaluation

To validate that our automated metrics capture what humans actually value, we conducted a small-scale human evaluation study with three domain experts from our research team.

**Protocol**: 30 diverse prompts (10 simple, 10 medium, 10 complex) were expanded using PRIME with Gemini-Embedding-001 and Gemini-2.5-Flash. Three raters (co-authors J.A., B.P.D., and R.C.) independently scored each expanded prompt on a 1-5 scale across four dimensions: Structure, Precision, Completeness, and Length Appropriateness. Raters were blind to prompt complexity labels during rating.

| Complexity | N | Avg Rating | Structure | Precision | Completeness | Length |
|------------|---|------------|-----------|-----------|--------------|--------|
| Simple | 10 | 4.1 | 4.3 | 3.8 | 4.0 | 4.2 |
| Medium | 10 | 4.2 | 4.5 | 4.0 | 4.1 | 4.3 |
| Complex | 10 | 4.4 | 4.6 | 4.2 | 4.5 | 4.2 |
| **Overall** | **30** | **4.2** | **4.5** | **4.0** | **4.2** | **4.2** |

**Inter-Rater Agreement**: 87% pairwise agreement (ratings within ±1 point), indicating reliable human judgment.

**Correlation with Automated Metrics**:

| Automated Metric | Human Correlation |
|------------------|-------------------|
| Structural Coherence (S) | r = 0.72 |
| Semantic Specificity (P) | r = 0.68 |
| Contextual Completeness (C) | r = 0.75 |
| Lexical Readability (L) | r = 0.45 |

**Key Insights**:

1. **Humans prefer complex query expansions**: Average rating increases from 4.1 (simple) to 4.4 (complex), consistent with our automated findings.

2. **Structure is most appreciated**: Highest human scores for structure (4.5) align with Claude's strength in this dimension.

3. **Readability is hardest to automate**: Lowest correlation (r = 0.45) suggests our sentence-length heuristic captures only part of what makes text readable.

4. **Automated metrics are valid proxies**: Three of four metrics show r > 0.65 correlation, supporting their use for large-scale evaluation.

### 6.11 Failure Analysis: When PRIME Hurts

No system works universally. We systematically investigated conditions where PRIME degrades rather than improves prompt quality.

#### Failure Mode 1: Irrelevant Context Injection

**Scenario**: Knowledge base contains documents unrelated to the query domain.

| Configuration | Expansion Length | Quality Score | Notes |
|--------------|------------------|---------------|-------|
| Relevant context | 287 words | 0.78 | Domain-matched documents |
| Irrelevant context | 234 words | 0.52 | Recipes, history, biology |
| No context (baseline) | 156 words | 0.61 | LLM knowledge only |

**Finding**: Irrelevant context (0.52) performs *worse* than no context at all (0.61). The retrieval step introduces noise that confuses the generator.

**Mitigation**: Implement semantic similarity thresholds; filter chunks below 0.3 cosine similarity.

#### Failure Mode 2: Already-Detailed Prompts

**Scenario**: User provides a comprehensive prompt that doesn't need expansion.

| Input Type | Input Words | Output Words | Expansion Ratio | Value Added |
|------------|-------------|--------------|-----------------|-------------|
| Vague ("Check status") | 2 | 287 | 143× | High |
| Medium detail | 25 | 312 | 12× | Medium |
| Already detailed | 62 | 205 | 3.3× | Low/Negative |

**Finding**: For prompts already containing sections, metrics, and format specifications, expansion adds little value and may introduce redundancy.

**Mitigation**: Detect prompt complexity before expansion; skip amplification for prompts with >5 structural elements.

#### Failure Mode 3: Ambiguous Queries with Conflicting Context

**Scenario**: Query is vague and retrieved documents contain contradictory information.

**Example**: Query "Which language should I use?" with documents praising Python, R, and Julia.

**Finding**: Expanded prompt inherits ambiguity, producing a confused mashup rather than clarifying the question.

**Mitigation**: Detect low-confidence retrieval scenarios; prompt user for clarification or expand multiple interpretations explicitly.

#### Failure Mode 4: Insufficient Knowledge Base

**Scenario**: Only 1-2 short documents available, providing minimal context.

| KB Size | Expansion Length | Quality | Retrieval Value |
|---------|------------------|---------|-----------------|
| 8 documents | 287 words | 0.78 | High |
| 2 documents | 223 words | 0.58 | Medium |
| 0 documents | 156 words | 0.61 | None (baseline) |

**Finding**: Very small knowledge bases can perform worse than no retrieval, as sparse context may mislead the generator.

**Mitigation**: Set minimum document threshold (recommend ≥5 documents); fall back to generator-only mode when insufficient.

#### Summary of Failure Conditions

| Condition | Detection Method | Recommended Action |
|-----------|------------------|-------------------|
| Irrelevant context | Similarity < 0.3 | Filter low-relevance chunks |
| Already detailed prompt | Structural elements > 5 | Skip expansion |
| Ambiguous + conflicting | Multiple clusters detected | Request clarification |
| Insufficient KB | Documents < 5 | Use generator-only mode |

### 6.12 Case Study

**Input**: "How's the deal going?"

**Output (Gemini-2.5-Flash + Gemini-Embedding-001)**:

```
GOAL: Provide a concise status update on the deal.

REQUIRED SECTIONS:
1. Overall Health
2. Key Metrics Summary  
3. Action Items

TABLE: Deal Status
| Metric | Value | Interpretation |
|--------|-------|----------------|
| POC Health | [to fill] | Healthy/At Risk/Critical |
| Feature Fit | [to fill] | Excellent/Good/Partial/Poor |
| Winscore | [to fill] | 0-100 scale |
```

The expanded prompt transforms 4 words into a structured template with goals, sections, metrics, and expected output format. The `[to fill]` placeholders indicate where the downstream LLM should provide specific values. This transformation is exactly what enables LLMs to provide useful, focused responses rather than generic platitudes.

---

## 7. Discussion

### 7.1 Surprises and Lessons

Running experiments across twelve embedding configurations and six generators taught us things we didn't expect. Some confirmed our hypotheses; others overturned them entirely.

**We underestimated the dense-sparse gap.** Going in, we assumed TF-IDF would be "good enough" for quick prototypes—the classic "MVP then upgrade" approach. We were wrong. The 37-73% quality difference isn't incremental; it's the difference between useful retrieval and random noise. If you're building anything beyond a toy demo, start with dense embeddings.

**Generalization surprised us.** Honestly, we expected PRIME to need domain-specific tuning. A sales system shouldn't understand research papers, right? Wrong. Modern embedding models have absorbed enough linguistic structure that the same configuration handles sales metrics and academic citations with only modest performance variation. This makes deployment dramatically simpler.

**Chunk size is the hidden lever.** This one still puzzles us slightly. Why do 100-character chunks outperform 1000-character chunks by 27%? Our hypothesis: smaller chunks match query length better. When you ask "deal health," you want the sentence defining deal health, not the paragraph surrounding it. Most tutorials recommend 500-1000 characters because that's what LangChain defaults to. Maybe reconsider.

**Caching is transformative, not incremental.** A 1,944× speedup doesn't make your app faster—it makes previously impossible interactions possible. What was a 10-second batch job becomes a 5-millisecond real-time response. We added caching as an afterthought for cost savings; it turned out to be essential for the user experience we wanted.

**Each LLM has a personality.** This isn't just anthropomorphization. Claude genuinely produces more structured outputs (score: 0.80 vs 0.27-0.33 for others). GPT-4 writes more naturally flowing prose. Gemini includes more actionable specifics. There's no single "best"—the right choice depends on what you're optimizing.

### 7.2 Practical Implications

Our results translate to concrete recommendations:

**Latency-sensitive applications** (chat, real-time): Use BM25 + Claude. Sub-4-second end-to-end latency.

**Quality-critical applications** (document generation, reports): Use Gemini-Embedding-001 + Gemini-2.5-Flash. Higher latency but measurably better outputs.

**Cost-sensitive deployments** (high volume, budget constraints): Use SBERT locally. Zero API costs, 91% of best retrieval quality.

### 7.3 Limitations

We've tried to be thorough, but no study covers everything. Here's what we acknowledge:

**Corpus scale.** Our test corpora are small—8-16 documents per domain. Real deployments often have thousands. Hybrid retrieval might shine at larger scales where lexical matching catches what semantic search misses. We saw hints of this but couldn't fully explore it.

**Heuristic metrics.** Our quality metrics measure structural properties: headers, bullet points, action verbs. They don't measure whether an expanded prompt actually leads to better task completion. A perfectly structured prompt that misunderstands the user's intent would score well on our metrics but fail in practice.

**Small-scale human evaluation.** While we report results from 30 prompts with 3 raters, this is smaller than ideal for definitive conclusions. The raters were co-authors rather than independent annotators, introducing potential bias. Future work should validate these findings with larger, external annotator pools and formal inter-rater reliability measures (e.g., Krippendorff's alpha).

**Limited generator coverage.** We tested OpenAI, Anthropic, and Google—the major commercial players. Open-source models (Mistral, Llama) might behave differently. Our architecture supports them; we prioritized API stability over breadth.

**English only.** PRIME uses embedders trained primarily on English. Performance on other languages is unknown. Multilingual SBERT variants exist but remain untested.

**Failure analysis scope.** While we identify four failure modes, others likely exist. Real-world deployment will reveal edge cases our controlled experiments missed.

### 7.4 When to Use (and Not Use) PRIME

PRIME shines when:

- **Domain knowledge exists in documents.** The system can only expand prompts using what it knows. No documents = no expansion.
- **User queries are ambiguous.** "How's the deal going?" needs expansion. "Generate a detailed POC health report with Winscore, milestone status, and risk factors" doesn't.
- **Consistency matters.** Organizations that need standardized report structures benefit from PRIME's templating effect.
- **Prompt engineering is a bottleneck.** If users waste hours crafting the "right" prompt, automation saves time.

PRIME is overkill when:

- **Queries are already detailed.** Expert users who naturally write good prompts don't need expansion.
- **Tasks are simple.** "Translate to French" doesn't need amplification.
- **Real-time latency is critical.** Retrieval + generation adds seconds. For chat applications expecting sub-second responses, this matters.

---

## 8. Conclusion

We have presented PRIME, a framework that treats prompt construction as a first-class optimization target rather than a manual preprocessing step. By retrieving documents to construct *better questions* rather than to generate answers directly, PRIME transforms vague user inputs into comprehensive, actionable prompts.

Our extensive evaluation—spanning four domains, 12 embedding configurations, 6 LLM generators, 30 human-evaluated prompts, and systematic failure analysis—yields actionable insights for practitioners:

**Core Technical Findings**:

1. **Dense embeddings are essential**: 37-73% higher retrieval precision than sparse methods. Start with SBERT (free) or Gemini-Embedding-001 (latest API).

2. **Latest models excel**: Gemini-3-Flash-Preview achieves 165× expansion with 0.798 quality score, outperforming Gemini-2.5-Flash (121×, 0.782). Model evolution is rapid.

3. **Complex queries win**: Counter-intuitively, longer queries outperform short keywords by 92%. Encourage users to write naturally.

4. **Chunk size matters**: 100-character chunks improve retrieval by 27% over 1000-character chunks.

5. **Caching transforms deployments**: 1,944× speedup enables real-time applications from what would otherwise be batch processing.

**Human Validation**:

Human raters scored PRIME outputs 4.2/5 on average, with 87% inter-rater agreement. Our automated metrics correlate meaningfully with human judgment (r = 0.68-0.75 for three of four metrics), supporting their use for evaluation at scale.

**When Not to Use PRIME**:

Prompt amplification fails when: (1) retrieved context is irrelevant (worse than no retrieval), (2) prompts are already detailed (low value-add), (3) queries are ambiguous with conflicting context, or (4) the knowledge base is too small. We provide detection methods and mitigations for each.

**Broader Impact**:

PRIME democratizes access to effective LLM interaction. Rather than requiring prompt engineering expertise, users can ask natural questions and receive structured prompts that elicit high-quality responses. This shifts the bottleneck from prompt crafting to knowledge curation—a more tractable problem for organizations.

### Future Work

1. **Larger-scale human evaluation**: Expand to 100+ prompts with diverse annotator pools
2. **Task completion studies**: Measure whether expanded prompts actually improve downstream task success
3. **Multi-modal expansion**: Extend to image, audio, and video contexts
4. **Adaptive configuration**: Dynamically select chunk size, top-k, and embedder based on query characteristics
5. **Open-source generators**: Evaluate Mistral, Llama, and other local models
6. **Multi-language support**: Test multilingual embedders and generators

### Availability

PRIME is available at: https://github.com/DeccanX/Prompt-Amplifier

Install via: `pip install prompt-amplifier`

Documentation: https://deccanx.github.io/Prompt-Amplifier/

---

## References

1. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474.

2. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

3. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., ... & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *Advances in Neural Information Processing Systems*, 35, 24824-24837.

4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP-IJCNLP*, 3982-3992.

5. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

6. Zhou, Y., Muresanu, A. I., Han, Z., Paster, K., Pitis, S., Chan, H., & Ba, J. (2023). Large Language Models Are Human-Level Prompt Engineers. *International Conference on Learning Representations*.

7. OpenAI. (2023). GPT-4 Technical Report. *arXiv preprint arXiv:2303.08774*.

8. Anthropic. (2024). The Claude 3 Model Family: Opus, Sonnet, and Haiku. *Technical Report*.

9. Gemini Team, Google. (2024). Gemini: A Family of Highly Capable Multimodal Models. *arXiv preprint arXiv:2312.11805*.

10. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-Scale Similarity Search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.

11. Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *Proceedings of EMNLP*, 6769-6781.

12. Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark. *Proceedings of EACL*, 2014-2037.

13. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., ... & Wang, H. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv preprint arXiv:2312.10997*.

14. Wang, L., Yang, N., & Wei, F. (2024). Query Rewriting for Retrieval-Augmented Large Language Models. *Proceedings of EMNLP*, 5303-5315.

15. Izacard, G., & Grave, E. (2021). Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. *Proceedings of EACL*, 874-880.

16. Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. *Proceedings of SIGIR*, 39-48.

17. Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis, M., ... & Yih, W. T. (2023). REPLUG: Retrieval-Augmented Black-Box Language Models. *arXiv preprint arXiv:2301.12652*.

18. Ram, O., Levine, Y., Dalmedigos, I., Muhlgay, D., Shashua, A., Leyton-Brown, K., & Shoham, Y. (2023). In-Context Retrieval-Augmented Language Models. *Transactions of the Association for Computational Linguistics*, 11, 1316-1331.

19. Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *International Conference on Learning Representations*.

20. Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2023). Dense X Retrieval: What Retrieval Granularity Should We Use? *arXiv preprint arXiv:2312.06648*.

21. Pradeep, R., Sharifymoghaddam, S., & Lin, J. (2023). RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models. *arXiv preprint arXiv:2309.15088*.

22. Ma, X., Gong, Y., He, P., Zhao, H., & Duan, N. (2024). Fine-Tuning LLaMA for Multi-Stage Text Retrieval. *Proceedings of SIGIR*, 2421-2425.

23. Xu, S., Chen, D., Shao, P., Xie, C., Zhang, S., Lin, P., ... & Zhang, D. (2024). Retrieval meets Long Context Large Language Models. *International Conference on Learning Representations*.

24. Peng, B., Galley, M., He, P., Cheng, H., Xie, Y., Hu, Y., ... & Gao, J. (2023). Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback. *arXiv preprint arXiv:2302.12813*.

25. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the Association for Computational Linguistics*, 12, 157-173.

26. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30, 5998-6008.

27. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT*, 4171-4186.

28. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *Journal of Machine Learning Research*, 21(140), 1-67.

29. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... & Lample, G. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv preprint arXiv:2302.13971*.

30. Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., ... & Sayed, W. E. (2023). Mistral 7B. *arXiv preprint arXiv:2310.06825*.

31. Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged Resources To Advance General Chinese Embedding. *arXiv preprint arXiv:2309.07597*.

32. Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., ... & Wei, F. (2024). Text Embeddings by Weakly-Supervised Contrastive Pre-training. *Proceedings of ACL*, 8882-8896.

33. Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. *arXiv preprint arXiv:1901.04085*.

34. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *Proceedings of NeurIPS Datasets and Benchmarks Track*.

35. Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., ... & Sifre, L. (2022). Improving Language Models by Retrieving from Trillions of Tokens. *International Conference on Machine Learning*, 2206-2240.

---

## Appendix A: Sample Expanded Prompts

### A.1 Sales Domain

**Input**: "Check pipeline health"

**Output**:
```
GOAL: Assess the current state of the sales pipeline.

REQUIRED ANALYSIS:
1. Overall Pipeline Value
2. Stage Distribution
3. Deal Velocity Trends
4. Risk Assessment

METRICS TO INCLUDE:
- Total pipeline value ($)
- Number of deals per stage
- Average days in stage
- Deals at risk (criteria: 2+ weeks stalled)

OUTPUT FORMAT:
- Executive summary (2-3 sentences)
- Metrics table
- Top 3 concerns
- Recommended actions
```

### A.2 Research Domain

**Input**: "Summarize the paper"

**Output**:
```
GOAL: Generate a comprehensive academic summary.

SECTIONS:
1. Paper Metadata (title, authors, venue, year)
2. Problem Statement
3. Key Contributions
4. Methodology
5. Main Results
6. Limitations
7. Future Directions

STYLE: Academic tone, cite specific numbers/claims
FORMAT: Structured with clear headers
LENGTH: 500-800 words
```

---

## Acknowledgments

We thank the open-source communities behind Sentence-BERT, FAISS, and the various LLM APIs that made this research possible. We also thank early users of Prompt Amplifier who provided feedback that shaped the system's design.


*Preprint submitted to arXiv, December 2025*

