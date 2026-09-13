# Retrieval-Augmented Generation (RAG) Systems

## 1. Overview and Problem Formulation
Retrieval-Augmented Generation (RAG) augments Large Language Models (LLMs) with an external non-parametric memory index. This eliminates hallucinations, enables real-time knowledge ingestion without costly fine-tuning, and provides verifiable citation provenance.

## 2. Core Architecture Pipeline
1. **Document Ingestion & Chunking**: Unstructured text is segmented using sentence-aware sliding windows with overlap to preserve syntactic boundaries.
2. **Dense Semantic Embedding**: Text chunks are projected into dense vector space (e.g. 384 dimensions) using transformer models or trained projection heads.
3. **Vector Indexing & Cosine Similarity**: Given query vector $q$ and document vector $d$, relevance is computed via cosine similarity:
   $$\text{Cosine}(q, d) = \frac{q \cdot d}{\|q\|_2 \|d\|_2}$$
4. **Top-K Retrieval**: High-similarity chunks are extracted and ranked.
5. **Context Injection & Synthesis**: Retrieved chunks are injected into a structured system prompt, and passed to the LLM (e.g. DeepSeek-R1, Meta Llama 3) for grounded response generation.
