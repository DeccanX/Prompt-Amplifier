# PRIME: A Modular Framework for Context-Aware Prompt Amplification Using Retrieval-Augmented Generation and Multi-Strategy Embedding

**Rajesh More**  
moreyrb@gmail.com

---

## Abstract

The effectiveness of Large Language Models (LLMs) hinges critically on prompt quality, yet crafting comprehensive prompts remains cognitively demanding. We introduce PRIME (Prompt Refinement via Information-driven Methods and Expansion), a modular framework that automatically transforms brief user inputs into semantically rich, well-structured prompts through retrieval-augmented generation. Our system implements a configurable pipeline with heterogeneous document loaders (10+ formats), pluggable embedding strategies (sparse and dense), persistent vector stores, built-in caching (1,391× speedup), and multi-provider LLM generators. We formalize prompt amplification as an information-theoretic optimization problem and introduce four evaluation metrics: structural coherence, semantic specificity, contextual completeness, and lexical readability. Comprehensive experiments across four domains (sales, research, support, content creation), five embedding configurations, and three LLM backends reveal that: (1) dense embeddings achieve 37-73% higher retrieval precision compared to sparse methods; (2) smaller chunk sizes (100 chars) improve retrieval by 27%; (3) Google's Gemini-2.0-flash achieves the highest prompt quality score (0.751); and (4) PRIME generalizes across domains without fine-tuning. PRIME is released as an open-source Python library (`pip install prompt-amplifier`), facilitating reproducible research in automated prompt engineering.

**Keywords:** Prompt Engineering, RAG, Retrieval-Augmented Generation, Text Embeddings, LLM, Prompt Amplification, Information Retrieval

---

## 1. Introduction

Over the past two years, Large Language Models have quietly revolutionized how millions of people work. Tools built on GPT-4, Claude, and Gemini now draft emails, summarize documents, and generate code across industries. Yet anyone who has spent time with these systems knows a frustrating truth: getting good results often requires surprisingly specific instructions.

Here's a scene that plays out daily in offices worldwide. A sales manager opens their AI assistant and types: "How's the deal going?" It's exactly what they'd ask a colleague. But the AI responds with a generic platitude about "monitoring key metrics" — useless for making an actual decision. The manager sighs, closes the tab, and goes back to manually reviewing spreadsheets.

The core problem is a mismatch. Humans communicate through context and shared understanding. We expect our colleagues to know what "the deal" means, which metrics matter, and how we prefer information presented. LLMs have none of this context unless we explicitly provide it. This *prompt engineering problem* has spawned an entire cottage industry of courses, consultants, and copy-paste template libraries.

### 1.1 The Problem We're Solving

We started with a simple question: what if we could automatically fill in the gaps that make prompts fail? When someone asks "How's the deal going?", the system should know — from the organization's own documents — that deals have Winscores, that there are specific health statuses, and that executive sponsor involvement matters. It should then construct a prompt asking about exactly those things.

This led us to an unconventional use of Retrieval-Augmented Generation. RAG systems typically retrieve documents to help answer questions. We retrieve documents to help *ask* better questions. When you type "deal status," PRIME finds your organization's definition of deal health, your metric categories, and your reporting preferences — then weaves these into a comprehensive prompt.

### 1.2 What We Contribute

This paper makes five main contributions:

1. **The PRIME Framework**: A modular, open-source library for RAG-based prompt amplification with pluggable components at every stage of the pipeline.

2. **Formal Problem Definition**: We cast prompt amplification as an optimization problem, providing theoretical grounding for what makes a "good" expanded prompt.

3. **Comprehensive Evaluation**: We benchmark 5 embedding strategies and 3 LLM generators, providing practitioners with concrete guidance on configuration choices.

4. **Novel Quality Metrics**: We introduce four metrics specifically designed to evaluate expanded prompts: structure, specificity, completeness, and readability.

5. **Open Implementation**: Everything we describe is available as a pip-installable package with full documentation.

---

## 2. Background and Related Work

### 2.1 Retrieval-Augmented Generation

RAG combines the knowledge stored in external documents with the generative capabilities of LLMs. The approach was formalized by Lewis et al. (2020), who showed that retrieving relevant passages before generation significantly improves performance on knowledge-intensive tasks.

The key insight is that LLMs have limitations in their parametric memory—they can't know everything, and their knowledge becomes stale. By retrieving relevant information at query time, RAG systems can provide accurate, up-to-date responses.

Our work extends this paradigm. Rather than using retrieval to answer questions, we use it to *formulate* better questions. The retrieved context informs what aspects of a topic are worth covering, what terminology is relevant, and what structure makes sense.

### 2.2 Prompt Engineering

The field has evolved through several phases:

**Manual Prompting**: Early work relied on hand-crafted templates. These work well for specific tasks but require expertise and don't generalize.

**Few-Shot Learning**: Brown et al. (2020) showed that providing examples in the prompt dramatically improves performance. This reduces the need for task-specific fine-tuning but still requires carefully selecting examples.

**Chain-of-Thought**: Wei et al. (2022) demonstrated that asking models to "think step by step" improves reasoning performance. This insight—that prompt structure affects output quality—motivates our focus on generating well-structured prompts.

**Automatic Optimization**: Recent work has explored using LLMs to optimize prompts themselves. Zhou et al. (2023) showed that LLMs can be surprisingly good prompt engineers. Our approach differs in that we leverage external knowledge rather than relying solely on the LLM's internal capabilities.

### 2.3 Text Embeddings

Effective retrieval requires representing text in a form suitable for similarity computation. Two main approaches exist:

**Sparse Representations**: Methods like TF-IDF and BM25 represent documents as high-dimensional vectors where each dimension corresponds to a vocabulary term. These are fast and interpretable but capture only lexical similarity.

**Dense Representations**: Neural approaches like Sentence-BERT (Reimers & Gurevych, 2019) map text to lower-dimensional continuous vectors that capture semantic meaning. These handle synonyms and paraphrasing but require more computation.

Our experiments compare both approaches, revealing significant quality differences that inform practical deployment decisions.

---

## 3. System Architecture

PRIME implements a five-stage pipeline: Ingestion → Chunking → Embedding → Retrieval → Generation. Each stage has a pluggable interface allowing customization.

### 3.1 Document Ingestion

The first challenge is getting data into the system. Real-world knowledge lives in diverse formats: PDFs, spreadsheets, web pages, even YouTube videos. PRIME supports 10+ formats through a unified loader interface:

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

Long documents must be split into manageable pieces for embedding and retrieval. We implement recursive chunking that respects natural boundaries:

```
Algorithm: RecursiveChunk(text, separators, size, overlap)
1. If text fits in chunk size, return [text]
2. Split by current separator
3. Combine adjacent pieces until size limit
4. Include overlap with previous chunk
5. Recurse with finer separators if needed
```

The algorithm first tries splitting by paragraphs, then sentences, then words, ensuring chunks are semantically coherent when possible.

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

We define prompt amplification formally:

**Definition (Prompt Amplification)**: Given input prompt *p*, knowledge corpus *K*, and quality function *Q*, find:

```
p* = argmax Q(p')
     p' ∈ P(p,K)
```

subject to: Intent(p') ≡ Intent(p)

In words: find the highest-quality expanded prompt that preserves the original intent while incorporating knowledge from the corpus.

### 4.2 Quality Metrics

We measure prompt quality along four dimensions:

**Structural Coherence (S)**: Does the prompt have clear organization? We detect headers, bullet points, numbered lists, and sections:

```
S(p) = (1/N) × Σ min(count(pattern_i, p) / threshold_i, 1)
```

**Semantic Specificity (P)**: Does the prompt give specific instructions? We check for action verbs ("generate", "analyze"), constraints ("must", "required"), and format specifications:

```
P(p) = (|ActionVerbs ∩ p| + |Constraints ∩ p| + |Formats ∩ p|) / MaxScore
```

**Contextual Completeness (C)**: Does the prompt cover expected sections? We check for goal, context, sections, instructions, and output format:

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

### 5.2 Configurations Tested

**Embedders**:
- TF-IDF (sparse, local)
- BM25 (sparse, local)
- Sentence-BERT MiniLM (dense, local)
- OpenAI text-embedding-3-small (dense, API)
- Google embedding (dense, API)

**Generators**:
- OpenAI GPT-4o-mini
- Anthropic Claude-3-Haiku
- Google Gemini-2.0-flash

### 5.3 Hardware

All experiments ran on an Apple M2 Pro with 32GB RAM. API calls used production endpoints with standard rate limits.

---

## 6. Results

### 6.1 Embedding Comparison

| Embedder | Dimension | Embed Time | Query Time | P@5 |
|----------|-----------|------------|------------|-----|
| TF-IDF | 431 | 3.2 ms | 0.11 ms | 0.45 |
| BM25 | 15 | 1.9 ms | 0.02 ms | 0.52 |
| SBERT-MiniLM | 384 | 6,256 ms | 35.8 ms | 0.71 |
| OpenAI-3-small | 1,536 | 972 ms | 2,676 ms | 0.78 |
| Google-embed | 768 | 1,055 ms | 298 ms | 0.76 |

**Key Findings**:

1. **Dense embeddings significantly outperform sparse**: P@5 improves from 0.45-0.52 (sparse) to 0.71-0.78 (dense), a 37-73% relative improvement.

2. **Local dense embeddings are viable**: SBERT achieves P@5 = 0.71 with no API cost, only 9% below the best API-based method.

3. **Sparse methods are dramatically faster**: Sub-millisecond query times vs. hundreds of milliseconds for dense methods. This matters for high-throughput applications.

### 6.2 Generator Comparison

| Generator | Model | Latency | Quality | Expansion |
|-----------|-------|---------|---------|-----------|
| OpenAI | gpt-4o-mini | 10.2s | 0.576 | 88× |
| Anthropic | claude-3-haiku | 3.3s | 0.687 | 43× |
| Google | gemini-2.0-flash | 3.9s | 0.751 | 125× |

**Key Findings**:

1. **Gemini achieves highest quality** (0.751), producing well-structured, specific prompts.

2. **Claude is fastest** at 3.3s, making it suitable for interactive applications.

3. **Expansion ratios vary widely**: From 43× (Claude) to 125× (Gemini). Higher isn't always better—it depends on whether you need comprehensive coverage or concise guidance.

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
| **Memory Cache** | 8.71 ms | 0.01 ms | **1,391×** | 50% |
| **Disk Cache** | 9.97 ms | 0.27 ms | 36.5× | 50% |
| No Cache | 8.96 ms | 8.96 ms | 1× | N/A |

**Key Benefits**:

1. **Dramatic speedup**: Memory caching provides 1,391× speedup for repeated queries, reducing response time from ~9ms to 0.01ms.

2. **Persistent caching**: Disk cache maintains speedup across sessions (36.5×), useful for production deployments.

3. **Cost savings**: For API-based embeddings/generators, caching eliminates redundant API calls, potentially saving 50%+ on API costs for applications with query repetition.

### 6.4 Case Study

**Input**: "How's the deal going?"

**Output (Gemini + SBERT)**:

```
GOAL: Provide a concise status update on the deal.

REQUIRED SECTIONS:
1. Overall Health
2. Key Metrics Summary  
3. Action Items

TABLE: Deal Status
| Metric | Value | Interpretation |
|--------|-------|----------------|
| POC Health | ? | Healthy/At Risk/Critical |
| Feature Fit | ? | Excellent/Good/Partial/Poor |
| Winscore | ? | 0-100 scale |
```

From 4 words to a structured prompt with goals, sections, and expected output format. This transformation is exactly what enables LLMs to provide useful, focused responses.

---

## 7. Discussion

### 7.1 Summary of Findings

Our comprehensive evaluation reveals several important insights:

1. **Dense embeddings are essential for quality**: Across all domains, semantic embeddings (SBERT, OpenAI, Google) consistently outperform lexical methods (TF-IDF, BM25) by 37-73% in retrieval precision.

2. **Domain generalization works**: PRIME achieves meaningful retrieval across all four tested domains, with scores ranging from 0.195 (support) to 0.519 (research). This suggests the framework can be deployed without domain-specific fine-tuning.

3. **Chunk size matters significantly**: Smaller chunks (100 chars) achieve 27% higher scores than larger chunks (1000 chars), highlighting the importance of this often-overlooked hyperparameter.

4. **Caching provides dramatic speedup**: Memory caching achieves 1,391× speedup for repeated queries, making PRIME suitable for interactive applications with query patterns.

5. **Generator strengths are complementary**: Claude excels at structure (0.80), GPT-4o at readability (1.00), and Gemini at specificity (0.25), suggesting value in task-specific generator selection.

### 7.2 Practical Implications

Our results provide concrete guidance for practitioners:

**For latency-sensitive applications**: Use sparse embeddings (BM25) with Claude. Total latency under 4 seconds.

**For quality-critical applications**: Use OpenAI embeddings with Gemini. Higher latency (5s+) but best output quality.

**For cost-sensitive deployments**: Use SBERT with any generator. No embedding API costs, 90% of best retrieval quality.

### 7.2 Limitations

Several limitations warrant acknowledgment:

1. **Single domain evaluation**: Our experiments focus on sales/POC data. Other domains may show different patterns.

2. **Quality metrics are heuristic**: Our metrics capture structural properties but don't directly measure task completion success.

3. **No human evaluation**: We rely on automated metrics. Human judgment of prompt quality would strengthen conclusions.

4. **Limited generator sample**: Testing only three generators leaves questions about others (Llama, Mistral, etc.).

### 7.3 When Prompt Amplification Helps

Our approach works best when:
- Users have domain knowledge available in documents
- Natural queries are ambiguous or underspecified
- Consistent, structured LLM outputs are desired
- The cost of poor prompts (irrelevant outputs) is high

It's less necessary when users already provide detailed prompts or when tasks are simple enough that brief prompts suffice.

---

## 8. Conclusion

We presented PRIME, a comprehensive framework for automatically transforming brief prompts into detailed, structured instructions through retrieval-augmented generation. Our extensive experiments across four domains, five embedding strategies, three LLM generators, and multiple ablation conditions reveal several important findings:

**Key Contributions and Results**:

1. **Embedding strategy matters**: Dense embeddings achieve 37-73% higher retrieval precision than sparse methods (P@5: 0.71-0.78 vs. 0.45-0.52), with local SBERT providing a strong cost-free alternative to API-based embeddings.

2. **Cross-domain generalization**: PRIME works across diverse domains (sales, research, support, content) without domain-specific tuning, with retrieval scores ranging from 0.195 to 0.519.

3. **Configuration insights**: Chunk sizes of 100-200 characters optimize retrieval precision (+27%), while k=3-5 provides the best balance of context breadth and relevance.

4. **Caching for production**: Our caching layer provides up to 1,391× speedup for repeated queries, essential for interactive applications.

5. **Complementary generators**: Claude excels at structure, GPT-4o at readability, and Gemini at specificity and overall quality (0.751).

**Impact**: PRIME reduces the expertise required for effective LLM interaction. Rather than learning prompt engineering techniques, users can simply ask natural questions and receive well-structured prompts that elicit high-quality responses. The framework is available as an open-source Python library with comprehensive documentation.

### Future Work

Several directions merit exploration:

1. **Multi-modal support**: Extending to images, audio, and video contexts for richer prompt generation
2. **Adaptive retrieval**: Dynamically adjusting top-k and chunk size based on query complexity and domain
3. **Fine-tuned generators**: Training specialized models for prompt expansion tasks
4. **Human evaluation**: Systematic user studies comparing human-written vs. PRIME-generated prompts
5. **Hybrid retrieval**: Combining sparse and dense methods for improved coverage
6. **Streaming generation**: Real-time prompt expansion for interactive applications
7. **Multi-language support**: Extending beyond English to multilingual prompt amplification

### Availability

PRIME is available at: https://github.com/DeccanX/Prompt-Amplifier

Install via: `pip install prompt-amplifier`

---

## References

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.

2. Brown, T., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.

3. Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*.

4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.

5. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in IR*.

6. Zhou, Y., et al. (2023). Large Language Models Are Human-Level Prompt Engineers. *ICLR*.

7. OpenAI. (2023). GPT-4 Technical Report. *arXiv*.

8. Anthropic. (2024). Claude 3 Model Card.

9. Google. (2024). Gemini: A Family of Highly Capable Multimodal Models.

10. Johnson, J., et al. (2019). Billion-scale Similarity Search with GPUs. *IEEE BigData*.

11. Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP*.

12. Muennighoff, N., et al. (2023). MTEB: Massive Text Embedding Benchmark. *EACL*.

13. Gao, L., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv:2312.10997*.

14. Wang, L., et al. (2024). Query Rewriting for Retrieval-Augmented Large Language Models. *EMNLP*.

15. Izacard, G., & Grave, E. (2021). Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. *EACL*.

16. Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. *SIGIR*.

17. Shi, W., et al. (2023). REPLUG: Retrieval-Augmented Black-Box Language Models. *arXiv:2301.12652*.

18. Ram, O., et al. (2023). In-Context Retrieval-Augmented Language Models. *TACL*.

19. Asai, A., et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *ICLR*.

20. Chen, J., et al. (2023). Dense X Retrieval: What Retrieval Granularity Should We Use? *arXiv:2312.06648*.

21. Pradeep, R., et al. (2023). RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models. *arXiv*.

22. Ma, X., et al. (2024). Fine-Tuning LLaMA for Multi-Stage Text Retrieval. *SIGIR*.

23. Xu, S., et al. (2024). Retrieval meets Long Context Large Language Models. *ICLR*.

24. Peng, B., et al. (2023). Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback. *arXiv*.

25. Liu, N., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. *TACL*.

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

*Paper submitted to arXiv, December 2024*

