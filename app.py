#!/usr/bin/env python3
"""
PyTorch RAG Engine: Web Dashboard & REST API.
"""

import os
import sys
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.rag_engine import RAGEngine

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_docs")
engine = RAGEngine()
if os.path.exists(DATA_DIR):
    engine.ingest_directory(DATA_DIR)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PyTorch RAG Engine — Neural Retrieval & LLM Grounding</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #0b0f19;
      --bg-surface: #111827;
      --bg-card: #1f2937;
      --border-subtle: #374151;
      --border-focus: #6366f1;
      --text-primary: #f9fafb;
      --text-secondary: #9ca3af;
      --text-muted: #6b7280;
      --accent-indigo: #6366f1;
      --accent-purple: #8b5cf6;
      --accent-emerald: #10b981;
      --accent-cyan: #06b6d4;
      --gradient-brand: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-base);
      color: var(--text-primary);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-subtle);
      padding: 1rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.85rem;
      text-decoration: none;
      color: inherit;
    }
    .brand-logo {
      width: 38px;
      height: 38px;
      border-radius: 10px;
      background: var(--gradient-brand);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 1.25rem;
      color: white;
      box-shadow: 0 0 16px rgba(99, 102, 241, 0.35);
    }
    .brand-title {
      font-weight: 700;
      font-size: 1.15rem;
      letter-spacing: -0.02em;
    }
    .badge {
      font-size: 0.7rem;
      padding: 0.2rem 0.6rem;
      background: rgba(99, 102, 241, 0.15);
      color: #818cf8;
      border: 1px solid rgba(99, 102, 241, 0.3);
      border-radius: 9999px;
      font-weight: 600;
      margin-left: 0.5rem;
    }

    .nav-actions {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .btn-secondary {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-primary);
      padding: 0.5rem 0.9rem;
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.15s ease-in-out;
      text-decoration: none;
    }
    .btn-secondary:hover {
      border-color: var(--accent-indigo);
      background: #283548;
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 2rem;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 2rem;
      flex: 1;
      width: 100%;
    }
    @media (max-width: 960px) {
      .container { grid-template-columns: 1fr; }
    }

    .sidebar {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      height: fit-content;
    }
    .section-title {
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .stats-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }
    .stat-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 0.85rem;
    }
    .stat-val {
      font-size: 1.4rem;
      font-weight: 700;
      color: var(--text-primary);
      font-family: 'JetBrains Mono', monospace;
    }
    .stat-lbl {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 0.2rem;
    }

    .doc-list {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      max-height: 260px;
      overflow-y: auto;
    }
    .doc-item {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 0.65rem 0.85rem;
      font-size: 0.82rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .doc-title {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 170px;
      font-weight: 500;
    }
    .chunk-tag {
      font-size: 0.7rem;
      color: var(--accent-cyan);
      background: rgba(6, 182, 212, 0.12);
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
    }

    .input-group {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      border-top: 1px solid var(--border-subtle);
      padding-top: 1.25rem;
    }
    .input-box {
      width: 100%;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      color: var(--text-primary);
      padding: 0.6rem 0.85rem;
      font-size: 0.85rem;
      font-family: inherit;
    }
    .input-box:focus {
      outline: none;
      border-color: var(--accent-indigo);
    }
    .btn-primary {
      background: var(--accent-indigo);
      border: none;
      color: white;
      padding: 0.65rem 1rem;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }
    .btn-primary:hover {
      background: #4f46e5;
    }

    .main-view {
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }

    .search-panel {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 1.75rem;
    }
    .presets {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 1.25rem;
    }
    .preset-pill {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 9999px;
      padding: 0.35rem 0.85rem;
      font-size: 0.78rem;
      color: var(--text-secondary);
      cursor: pointer;
      transition: all 0.15s;
    }
    .preset-pill:hover {
      border-color: var(--accent-indigo);
      color: var(--text-primary);
      background: rgba(99, 102, 241, 0.1);
    }

    .search-row {
      position: relative;
      display: flex;
      align-items: center;
    }
    .search-bar {
      width: 100%;
      background: var(--bg-card);
      border: 1.5px solid var(--border-subtle);
      border-radius: 10px;
      padding: 0.95rem 1.25rem;
      padding-right: 120px;
      font-size: 0.98rem;
      color: var(--text-primary);
      font-family: inherit;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .search-bar:focus {
      outline: none;
      border-color: var(--accent-indigo);
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }
    .submit-btn {
      position: absolute;
      right: 8px;
      background: var(--gradient-brand);
      color: white;
      border: none;
      padding: 0.6rem 1.15rem;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.85rem;
      cursor: pointer;
    }

    .results-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 1.75rem;
      display: none;
      flex-direction: column;
      gap: 1.25rem;
      animation: fadeIn 0.25s ease-in-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .results-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 0.85rem;
    }
    .latency-badge {
      font-size: 0.75rem;
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent-emerald);
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.25);
      padding: 0.25rem 0.65rem;
      border-radius: 6px;
    }
    .answer-box {
      font-size: 0.95rem;
      line-height: 1.7;
      color: #e5e7eb;
      white-space: pre-wrap;
    }

    .citations-header {
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      border-top: 1px solid var(--border-subtle);
      padding-top: 1.25rem;
      margin-bottom: 0.75rem;
    }
    .citations-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1rem;
    }
    .citation-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .citation-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .citation-name {
      font-weight: 600;
      font-size: 0.82rem;
      color: #93c5fd;
    }
    .citation-score {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      background: rgba(99, 102, 241, 0.15);
      color: #818cf8;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
    }
    .citation-text {
      font-size: 0.78rem;
      line-height: 1.5;
      color: var(--text-secondary);
      font-style: italic;
    }

    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(6px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 200;
    }
    .modal-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      width: 100%;
      max-width: 480px;
      padding: 2rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }
    .modal-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .modal-close-btn {
      background: none;
      border: none;
      color: var(--text-muted);
      font-size: 1.5rem;
      cursor: pointer;
    }
  </style>
</head>
<body>

  <header>
    <a href="/" class="brand">
      <div class="brand-logo">⚡</div>
      <div class="brand-title">PyTorch RAG Engine</div>
      <span class="badge">Dual-Engine</span>
    </a>
    <div class="nav-actions">
      <button class="btn-secondary" onclick="openSettings()">
        ⚙️ Settings
      </button>
      <a href="https://github.com/ozodbek-bosimov/pytorch-rag-engine" target="_blank" class="btn-secondary">
        ⭐ GitHub
      </a>
    </div>
  </header>

  <main class="container">
    <aside class="sidebar">
      <div class="section-title">
        <span>Knowledge Base</span>
        <span id="doc-badge" class="chunk-tag">0 Docs</span>
      </div>

      <div class="stats-row">
        <div class="stat-card">
          <div id="stat-chunks" class="stat-val">0</div>
          <div class="stat-lbl">Indexed Chunks</div>
        </div>
        <div class="stat-card">
          <div id="stat-dim" class="stat-val">384</div>
          <div class="stat-lbl">Vector Dim</div>
        </div>
      </div>

      <div class="doc-list" id="doc-list"></div>

      <div class="input-group">
        <div class="section-title">
          <span>Ingest Document</span>
        </div>
        <input type="text" id="ingest-title" class="input-box" placeholder="Document title...">
        <textarea id="ingest-content" class="input-box" rows="4" placeholder="Document content or markdown..."></textarea>
        <button class="btn-primary" onclick="ingestDocument()">
          📥 Add to Vector Store
        </button>
      </div>
    </aside>

    <section class="main-view">
      <div class="search-panel">
        <div class="presets">
          <div class="preset-pill" onclick="applyQuery('What are the five essential steps of a standard PyTorch training loop?')">
            🔥 PyTorch Training Loop (5 Steps)
          </div>
          <div class="preset-pill" onclick="applyQuery('What is RAG and how does it prevent LLM hallucination?')">
            🧠 RAG Architecture & Grounding
          </div>
          <div class="preset-pill" onclick="applyQuery('What are the benefits of weight quantization and SIMD inference for deployment?')">
            ⚡ Quantization & CPU Inference
          </div>
        </div>

        <div class="search-row">
          <input 
            type="text" 
            id="query-input" 
            class="search-bar" 
            placeholder="Ask a technical question across the indexed corpus..."
            onkeydown="if(event.key === 'Enter') submitQuery()"
          >
          <button id="search-btn" class="submit-btn" onclick="submitQuery()">
            Search ↵
          </button>
        </div>
      </div>

      <div id="results-card" class="results-card">
        <div class="results-top">
          <div style="display:flex; align-items:center; gap:0.6rem;">
            <span style="font-weight:700;">Answer:</span>
            <span id="model-name" style="font-size:0.75rem; color:var(--text-secondary);"></span>
          </div>
          <div id="latency-badge" class="latency-badge"></div>
        </div>

        <div id="answer-body" class="answer-box"></div>

        <div>
          <div class="citations-header">📚 Citation Provenance & Similarity Rankings</div>
          <div id="citations-grid" class="citations-grid"></div>
        </div>
      </div>
    </section>
  </main>

  <div id="settings-modal" class="modal-backdrop">
    <div class="modal-card">
      <div class="modal-head">
        <h3 style="font-size:1.1rem; font-weight:700;">API & Model Settings</h3>
        <button class="modal-close-btn" onclick="closeSettings()">&times;</button>
      </div>
      <p style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
        Configure OpenRouter API credentials to enable online LLM generation with models like DeepSeek-R1 and Llama 3.3.
      </p>
      <div>
        <label style="display:block; font-size:0.8rem; font-weight:600; margin-bottom:0.4rem;">OpenRouter API Key:</label>
        <input type="password" id="api-key-input" class="input-box" placeholder="sk-or-v1-...">
      </div>
      <div>
        <label style="display:block; font-size:0.8rem; font-weight:600; margin-bottom:0.4rem;">Target Model:</label>
        <select id="model-select" class="input-box">
          <option value="deepseek/deepseek-r1:free">DeepSeek-R1 (Free Reasoning Model)</option>
          <option value="meta-llama/llama-3.3-70b-instruct:free">Meta Llama 3.3 70B Instruct (Free)</option>
          <option value="google/gemini-2.0-flash-exp:free">Google Gemini 2.0 Flash (Free)</option>
        </select>
      </div>
      <button class="btn-primary" onclick="saveSettings()">
        Save Settings
      </button>
    </div>
  </div>

  <script>
    async function refreshStats() {
      try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        document.getElementById('stat-chunks').textContent = data.total_chunks;
        document.getElementById('stat-dim').textContent = data.embedding_dimension;
        document.getElementById('doc-badge').textContent = data.total_documents + ' Docs';

        const listEl = document.getElementById('doc-list');
        listEl.innerHTML = '';
        data.indexed_documents.forEach(doc => {
          const div = document.createElement('div');
          div.className = 'doc-item';
          div.innerHTML = `
            <span class="doc-title" title="${doc.title}">📄 ${doc.title}</span>
            <span class="chunk-tag">${doc.chunk_count} chk</span>
          `;
          listEl.appendChild(div);
        });
      } catch (err) {
        console.error('Stats error:', err);
      }
    }

    function applyQuery(query) {
      document.getElementById('query-input').value = query;
      submitQuery();
    }

    async function submitQuery() {
      const input = document.getElementById('query-input');
      const query = input.value.trim();
      if (!query) return;

      const btn = document.getElementById('search-btn');
      const card = document.getElementById('results-card');
      const answerEl = document.getElementById('answer-body');
      const latencyEl = document.getElementById('latency-badge');
      const modelEl = document.getElementById('model-name');
      const grid = document.getElementById('citations-grid');

      btn.textContent = 'Searching...';
      btn.disabled = true;

      try {
        const savedModel = localStorage.getItem('rag_model') || 'deepseek/deepseek-r1:free';
        const res = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: query, model: savedModel })
        });
        const data = await res.json();

        card.style.display = 'flex';
        answerEl.textContent = data.answer;
        modelEl.textContent = data.model_used + (data.fallback_used ? ' (Offline Mode)' : '');
        latencyEl.textContent = `⚡ Retrieval: ${data.retrieval_time_ms}ms | Total: ${data.total_latency_ms}ms`;

        grid.innerHTML = '';
        if (data.sources && data.sources.length > 0) {
          data.sources.forEach((s, idx) => {
            const cardEl = document.createElement('div');
            cardEl.className = 'citation-card';
            cardEl.innerHTML = `
              <div class="citation-top">
                <span class="citation-name">#${idx+1} ${s.document_title}</span>
                <span class="citation-score">${(s.similarity_score * 100).toFixed(1)}% match</span>
              </div>
              <div class="citation-text">"${s.text}"</div>
            `;
            grid.appendChild(cardEl);
          });
        } else {
          grid.innerHTML = '<div style="color:var(--text-muted); font-size:0.85rem;">No citations found.</div>';
        }
      } catch (err) {
        alert('Query error: ' + err);
      } finally {
        btn.textContent = 'Search ↵';
        btn.disabled = false;
      }
    }

    async function ingestDocument() {
      const title = document.getElementById('ingest-title').value.trim();
      const content = document.getElementById('ingest-content').value.trim();
      if (!title || !content) {
        alert('Please provide both title and content.');
        return;
      }

      try {
        const res = await fetch('/api/ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: title, content: content })
        });
        const data = await res.json();
        alert(`Ingestion complete! ${data.chunks_created} chunks added.`);
        document.getElementById('ingest-title').value = '';
        document.getElementById('ingest-content').value = '';
        refreshStats();
      } catch (err) {
        alert('Ingestion error: ' + err);
      }
    }

    function openSettings() {
      document.getElementById('settings-modal').style.display = 'flex';
      document.getElementById('api-key-input').value = localStorage.getItem('openrouter_key') || '';
      document.getElementById('model-select').value = localStorage.getItem('rag_model') || 'deepseek/deepseek-r1:free';
    }

    function closeSettings() {
      document.getElementById('settings-modal').style.display = 'none';
    }

    async function saveSettings() {
      const key = document.getElementById('api-key-input').value.trim();
      const model = document.getElementById('model-select').value;
      localStorage.setItem('openrouter_key', key);
      localStorage.setItem('rag_model', model);

      if (key) {
        await fetch('/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ api_key: key, model: model })
        });
      }
      closeSettings();
      alert('Settings saved successfully.');
    }

    window.addEventListener('DOMContentLoaded', refreshStats);
  </script>
</body>
</html>
"""


class RAGRequestHandler(BaseHTTPRequestHandler):

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._set_headers(200, "text/html; charset=utf-8")
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if path == "/api/stats":
            stats = engine.get_system_stats()
            self._set_headers(200)
            self.wfile.write(json.dumps(stats).encode("utf-8"))
            return

        if path == "/api/health":
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "healthy",
                "indexed_documents": len(engine.indexed_docs),
                "indexed_chunks": engine.vector_store.count()
            }).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Not Found"}).encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"

        try:
            body = json.loads(post_body)
        except Exception:
            body = {}

        if path == "/api/query":
            query_text = body.get("query", "").strip()
            custom_model = body.get("model", None)
            top_k = int(body.get("top_k", 3))

            if not query_text:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Empty query"}).encode("utf-8"))
                return

            res = engine.query(query_text, top_k=top_k, custom_model=custom_model)
            self._set_headers(200)
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        if path == "/api/ingest":
            title = body.get("title", "Document").strip()
            content = body.get("content", "").strip()
            doc_id = title.lower().replace(" ", "_") + ".txt"

            if not content:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Document content is empty"}).encode("utf-8"))
                return

            chunks_cnt = engine.ingest_text(doc_id, title, content)
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "success",
                "document_id": doc_id,
                "chunks_created": chunks_cnt
            }).encode("utf-8"))
            return

        if path == "/api/settings":
            api_key = body.get("api_key")
            model = body.get("model")
            if api_key:
                engine.llm_client.api_key = api_key
            if model:
                engine.llm_client.model = model
            self._set_headers(200)
            self.wfile.write(json.dumps({"status": "settings_updated"}).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Not Found"}).encode("utf-8"))


def run_server(port: int = 8085):
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, RAGRequestHandler)
    print(f"[*] PyTorch RAG Engine Server running on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Server stopped.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8085
    run_server(port)
