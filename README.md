# Neural RAG Engine

A lightweight Retrieval-Augmented Generation (RAG) system with dense vector retrieval, in-memory indexing, and LLM grounding.

**Live Demo:** [https://neural-rag.alwaysdata.net](https://neural-rag.alwaysdata.net)

---

## Features

- **Dense Retrieval:** Sub-millisecond vector similarity search over document chunks.
- **Sentence-Aware Chunking:** Smart text splitting that preserves sentence boundaries.
- **LLM Grounding & Citations:** Answers grounded directly in retrieved context with source attribution.
- **Zero-Dependency Core:** Built using standard Python libraries, lightweight and fast.
- **Web UI & REST API:** Minimalist dark dashboard with document upload (PDF, Markdown, TXT) and search.

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/ozodbek-bosimov/neural-rag-engine.git
cd neural-rag-engine
pip install -r requirements.txt
```

### 2. Run Locally

```bash
python3 app.py 8085
```

Open [http://localhost:8085](http://localhost:8085) in your browser.

### 3. Run Tests

```bash
python3 -m unittest tests/test_pipeline.py
python3 evaluate.py
```

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web dashboard |
| `GET` | `/api/stats` | Document and vector index statistics |
| `GET` | `/api/health` | System health check |
| `POST` | `/api/query` | Search and generate grounded answer (`{ "query": "..." }`) |
| `POST` | `/api/upload` | Upload and index a document (PDF, TXT, MD) |
| `POST` | `/api/delete` | Delete a document from the current session |

---

[MIT License](LICENSE)
