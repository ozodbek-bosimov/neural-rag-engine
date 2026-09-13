# Neural RAG Engine

A lightweight, production-ready Retrieval-Augmented Generation (RAG) system with a custom neural embedding projection head, dense cosine retrieval, and OpenRouter LLM integration.

---

## Features

- **Neural Projection Head (`nn.Module`)**: Custom 2-layer projection head (`Linear -> GELU -> Dropout -> Linear -> LayerNorm -> L2 Normalize`) trained via `CosineEmbeddingLoss` and `AdamW`.
- **Sentence-Aware Chunker**: Sliding window chunker with configurable token overlap preserving sentence and paragraph boundaries.
- **In-Memory Dense Vector Store**: Sub-millisecond cosine similarity ranking over normalized document vectors.
- **Dual Synthesis Engine**: Online generative synthesis via OpenRouter (DeepSeek-R1, Meta Llama 3.3, Google Gemini) with deterministic offline semantic extraction fallback.
- **Evaluation Suite**: Automated benchmark computing **Hit Rate @ 1**, **Hit Rate @ K**, and **Mean Reciprocal Rank (MRR)**.
- **Modern Dark UI & REST API**: Native zero-dependency HTTP server with interactive search dashboard, citation provenance cards, and REST endpoints.

---

## Architecture

```
[Raw Documents] (.md, .txt)
       │
       ▼
[DocumentChunker] (Sentence-aware sliding window + overlap)
       │
       ▼
[Dense Embedding & Neural Projection Head] (L2 normalized ℝ^256)
       │
       ▼
[VectorStore Index] (In-memory cosine similarity search)
       │
       ▼  ◄── [User Query]
[Top-K Retrieval]
       │
       ▼
[Grounded Prompt Injection]
       │
       ▼
[OpenRouter LLM / Offline Synthesizer]
       │
       ▼
[Answer + Citation Provenance + Latency Metrics]
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/ozodbek-bosimov/neural-rag-engine.git
cd neural-rag-engine
pip install -r requirements.txt
```

### 2. Run the Web Dashboard & REST API

```bash
python3 app.py 8085
```

Open `http://localhost:8085` in your browser.

### 3. Run Benchmark Evaluation

```bash
python3 evaluate.py
```

Sample Benchmark Output:
```
====================================================================
RAG RETRIEVAL BENCHMARK (Hit Rate @ K, MRR, Latency)
====================================================================
[PASS] Test 1: 'How does PyTorch autograd and reverse-mo...' -> Top-1: pytorch_deep_learning_guide.md (0.19 ms)
[PASS] Test 2: 'What are the five essential steps of a s...' -> Top-1: pytorch_deep_learning_guide.md (0.17 ms)
[PASS] Test 3: 'What is RAG and how does it prevent LLM ...' -> Top-1: retrieval_augmented_generation.md (0.14 ms)
[PASS] Test 4: 'What mathematical formula defines vector...' -> Top-1: retrieval_augmented_generation.md (0.15 ms)
[PASS] Test 5: 'How do Conv2D kernels and max pooling la...' -> Top-1: convolutional_neural_networks.md (0.17 ms)
[PASS] Test 6: 'What are the benefits of weight quantiza...' -> Top-1: convolutional_neural_networks.md (0.15 ms)
--------------------------------------------------------------------
BENCHMARK METRICS:
  • Hit Rate @ 1: 100.0%
  • Hit Rate @ 3: 100.0%
  • MRR (Mean Reciprocal Rank): 1.0000 / 1.0000
  • Mean Latency: 0.16 ms
====================================================================
```

### 4. Run Unit Tests

```bash
python3 -m unittest tests/test_pipeline.py
```

---

## REST API Reference

### Query Endpoint
`POST /api/query`

**Payload:**
```json
{
  "query": "What are the five essential steps of a standard PyTorch training loop?",
  "top_k": 3,
  "model": "deepseek/deepseek-r1:free"
}
```

**Response:**
```json
{
  "query": "What are the five essential steps of a standard PyTorch training loop?",
  "answer": "A standard supervised training iteration consists of five deterministic steps: 1. optimizer.zero_grad() ...",
  "sources": [
    {
      "document_id": "pytorch_deep_learning_guide.md",
      "document_title": "Pytorch Deep Learning Guide",
      "chunk_id": "pytorch_deep_learning_guide.md_c0",
      "similarity_score": 0.2851,
      "text": "..."
    }
  ],
  "retrieval_time_ms": 0.17,
  "generation_time_ms": 0.0,
  "total_latency_ms": 0.17,
  "model_used": "Offline Extractive Synthesizer",
  "fallback_used": true
}
```

### Ingestion Endpoint
`POST /api/ingest`

**Payload:**
```json
{
  "title": "Attention Mechanisms",
  "content": "Scaled dot-product attention computes compatibility between queries and keys..."
}
```

### System Health & Stats
- `GET /api/health`
- `GET /api/stats`

---

## Project Structure

```
neural-rag-engine/
├── app.py                      # HTTP server, REST API, and dashboard
├── evaluate.py                 # Hit Rate@K and MRR evaluation suite
├── requirements.txt            # Dependency list
├── Dockerfile                  # Container definition
├── README.md                   # Technical documentation
├── data/
│   └── sample_docs/            # Technical domain corpus
│       ├── convolutional_neural_networks.md
│       ├── pytorch_deep_learning_guide.md
│       └── retrieval_augmented_generation.md
├── src/
│   ├── __init__.py
│   ├── chunker.py              # Sentence-preserving sliding window chunker
│   ├── dataset.py              # PyTorch Dataset for paired contrastive data
│   ├── llm_client.py           # OpenRouter client & offline fallback
│   ├── model.py                # Neural EmbeddingProjectionHead (nn.Module)
│   ├── rag_engine.py           # RAG orchestrator & vector index
│   ├── train.py                # Neural training loop (AdamW + CosineLoss)
│   └── vector_store.py         # Dense vector store with cosine ranking
└── tests/
    ├── __init__.py
    └── test_pipeline.py        # Automated test suite
```

---

## License

MIT
