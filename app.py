#!/usr/bin/env python3
"""
Neural RAG Engine: Web Dashboard & REST API.
"""

import os
import sys
import json
import time
import base64
import threading
from typing import Dict, Any, Optional
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.rag_engine import RAGEngine

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_docs")

# Load .env file if present
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# Multi-session state management
# Each browser load/refresh generates an isolated in-memory session initialized with sample documents.
session_lock = threading.Lock()
sessions: Dict[str, Dict[str, Any]] = {}
GLOBAL_API_KEY = os.getenv("GEMINI_API_KEY")
GLOBAL_MODEL = "gemini-flash-lite-latest"
MAX_SESSIONS = 100
SESSION_TTL = 3600  # 1 hour

def cleanup_old_sessions():
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s["last_active"] > SESSION_TTL]
    for sid in expired:
        del sessions[sid]
    if len(sessions) > MAX_SESSIONS:
        sorted_s = sorted(sessions.items(), key=lambda x: x[1]["last_active"])
        for sid, _ in sorted_s[: len(sessions) - MAX_SESSIONS]:
            del sessions[sid]

def get_engine_for_session(session_id: str) -> RAGEngine:
    with session_lock:
        cleanup_old_sessions()
        sid = (session_id or "default").strip()
        if sid in sessions:
            sessions[sid]["last_active"] = time.time()
            return sessions[sid]["engine"]
        
        # Fresh isolated session populated with default sample documents
        eng = RAGEngine(api_key=GLOBAL_API_KEY, model_name=GLOBAL_MODEL)
        if os.path.exists(DATA_DIR):
            eng.ingest_directory(DATA_DIR)
        
        sessions[sid] = {
            "engine": eng,
            "last_active": time.time()
        }
        return eng


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Neural RAG Engine — Dense Retrieval & LLM Grounding</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root {
      --bg-base: #0d1117;
      --bg-surface: #161b22;
      --bg-card: #21262d;
      --border-subtle: #30363d;
      --border-focus: #58a6ff;
      --text-primary: #f0f6fc;
      --text-secondary: #8b949e;
      --text-muted: #6e7681;
      --accent-blue: #2f81f7;
      --accent-green: #238636;
      --accent-green-hover: #2ea043;
      --accent-tag: #1f6feb;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background-color: var(--bg-base);
      color: var(--text-primary);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-subtle);
      padding: 0.85rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .brand {
      display: flex;
      align-items: baseline;
      gap: 0.75rem;
      text-decoration: none;
      color: inherit;
    }
    .brand-title {
      font-weight: 700;
      font-size: 1.05rem;
      letter-spacing: -0.01em;
      color: var(--text-primary);
    }
    .brand-sub {
      font-size: 0.75rem;
      color: var(--text-secondary);
      font-weight: 400;
    }

    .nav-actions {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }
    .btn-outline {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-primary);
      padding: 0.45rem 0.85rem;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.45rem;
      text-decoration: none;
      transition: background 0.15s, border-color 0.15s;
    }
    .btn-outline:hover {
      background: #30363d;
      border-color: #8b949e;
    }
    .btn-github {
      background: #21262d;
      border: 1px solid var(--border-subtle);
      color: #c9d1d9;
      padding: 0.45rem 0.85rem;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.45rem;
      text-decoration: none;
      transition: background 0.15s, border-color 0.15s, color 0.15s;
    }
    .btn-github:hover {
      background: #30363d;
      border-color: #8b949e;
      color: #f0f6fc;
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 1.75rem 2rem;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 1.75rem;
      flex: 1;
      width: 100%;
    }
    @media (max-width: 960px) {
      .container { grid-template-columns: 1fr; padding: 1rem; }
    }

    .sidebar {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      height: fit-content;
    }
    .section-title {
      font-size: 0.75rem;
      font-weight: 600;
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
      gap: 0.6rem;
    }
    .stat-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 0.65rem 0.85rem;
      text-align: center;
    }
    .stat-val {
      font-size: 1.35rem;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-primary);
    }
    .stat-lbl {
      font-size: 0.68rem;
      color: var(--text-secondary);
      margin-top: 0.15rem;
    }

    .doc-list {
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
      max-height: 240px;
      overflow-y: auto;
    }
    .doc-item {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 0.55rem 0.75rem;
      font-size: 0.8rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.5rem;
    }
    .doc-title {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      flex: 1;
      min-width: 0;
      font-weight: 500;
      color: var(--text-primary);
      display: flex;
      align-items: center;
    }
    .chunk-tag {
      font-size: 0.68rem;
      color: #58a6ff;
      background: rgba(56, 139, 253, 0.1);
      border: 1px solid rgba(56, 139, 253, 0.2);
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      font-family: 'JetBrains Mono', monospace;
      white-space: nowrap;
      flex-shrink: 0;
    }
    .delete-btn {
      background: transparent;
      border: 1px solid transparent;
      cursor: pointer;
      padding: 1px 5px;
      border-radius: 4px;
      font-size: 0.8rem;
      line-height: 1;
      color: var(--text-muted);
      transition: all 0.15s ease;
    }
    .delete-btn:hover {
      color: #f85149;
      background: rgba(248, 81, 73, 0.12);
      border-color: rgba(248, 81, 73, 0.3);
    }
    .reset-btn {
      background: transparent;
      border: 1px solid var(--border-subtle);
      cursor: pointer;
      padding: 1px 6px;
      border-radius: 4px;
      font-size: 0.75rem;
      line-height: 1.2;
      color: var(--text-muted);
      transition: all 0.15s ease;
    }
    .reset-btn:hover {
      color: var(--text-primary);
      border-color: #8b949e;
      background: var(--bg-card);
    }

    .input-group {
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }
    .ingest-tabs {
      display: flex;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 2px;
      gap: 2px;
      margin-bottom: 0.2rem;
    }
    .ingest-tab {
      flex: 1;
      background: transparent;
      border: none;
      color: var(--text-secondary);
      padding: 0.35rem 0.6rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 500;
      cursor: pointer;
      font-family: inherit;
      transition: all 0.15s ease;
      text-align: center;
    }
    .ingest-tab:hover {
      color: var(--text-primary);
    }
    .ingest-tab.active {
      background: var(--bg-surface);
      color: var(--text-primary);
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
      font-weight: 600;
    }
    .input-box {
      width: 100%;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      color: var(--text-primary);
      padding: 0.55rem 0.75rem;
      font-size: 0.82rem;
      font-family: inherit;
    }
    .input-box:focus {
      outline: none;
      border-color: var(--border-focus);
    }
    .btn-primary {
      background: var(--accent-green);
      border: 1px solid rgba(240, 246, 252, 0.1);
      color: white;
      padding: 0.55rem 0.85rem;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }
    .btn-primary:hover {
      background: var(--accent-green-hover);
    }

    .ingest-panel {
      height: 156px;
      box-sizing: border-box;
    }
    .upload-dropzone {
      height: 156px;
      border: 1px dashed var(--border-subtle);
      border-radius: 6px;
      padding: 1rem 0.85rem;
      text-align: center;
      cursor: pointer;
      background: rgba(33, 38, 45, 0.5);
      transition: all 0.15s ease;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 0.35rem;
      box-sizing: border-box;
    }
    .upload-dropzone:hover, .upload-dropzone.dragover {
      border-color: var(--border-focus);
      background: rgba(56, 139, 253, 0.05);
    }
    .upload-title {
      font-size: 0.82rem;
      font-weight: 500;
      color: var(--text-primary);
    }
    .upload-subtitle {
      font-size: 0.7rem;
      color: var(--text-muted);
    }
    #ingest-text-view {
      height: 156px;
      box-sizing: border-box;
      flex-direction: column;
      gap: 0.45rem;
    }
    #ingest-text-view .input-box#ingest-title {
      height: 32px;
      padding: 0.45rem 0.65rem;
      font-size: 0.8rem;
      box-sizing: border-box;
      flex-shrink: 0;
    }
    #ingest-text-view textarea#ingest-content {
      flex: 1;
      min-height: 0;
      resize: none;
      padding: 0.45rem 0.65rem;
      font-size: 0.8rem;
      box-sizing: border-box;
      font-family: inherit;
    }
    #ingest-text-view .btn-primary {
      height: 32px;
      padding: 0.45rem 0.65rem;
      font-size: 0.8rem;
      box-sizing: border-box;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .upload-status {
      font-size: 0.75rem;
      padding: 0.45rem 0.65rem;
      border-radius: 4px;
      display: none;
      line-height: 1.4;
    }
    .upload-status.success {
      display: block;
      background: rgba(46, 160, 67, 0.12);
      color: #3fb950;
      border: 1px solid rgba(46, 160, 67, 0.25);
    }
    .upload-status.loading {
      display: block;
      background: rgba(56, 139, 253, 0.12);
      color: #58a6ff;
      border: 1px solid rgba(56, 139, 253, 0.25);
    }
    .upload-status.error {
      display: block;
      background: rgba(248, 81, 73, 0.12);
      color: #f85149;
      border: 1px solid rgba(248, 81, 73, 0.25);
    }

    .main-view {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .search-panel {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 1.25rem;
    }
    .demo-queries {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.45rem;
      margin-top: 0.85rem;
    }
    .demo-label {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-right: 0.15rem;
    }
    .demo-btn {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 4px;
      padding: 0.25rem 0.55rem;
      font-size: 0.75rem;
      color: var(--text-secondary);
      cursor: pointer;
      font-family: inherit;
      transition: all 0.15s ease;
    }
    .demo-btn:hover {
      color: var(--text-primary);
      border-color: #8b949e;
      background: #30363d;
    }

    .search-row {
      position: relative;
      display: flex;
      align-items: center;
    }
    .search-bar {
      width: 100%;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 0.75rem 1rem;
      padding-right: 95px;
      font-size: 0.92rem;
      color: var(--text-primary);
      font-family: inherit;
      transition: border-color 0.15s, box-shadow 0.15s;
    }
    .search-bar:focus {
      outline: none;
      border-color: var(--border-focus);
      box-shadow: 0 0 0 3px rgba(56, 139, 253, 0.15);
    }
    .submit-btn {
      position: absolute;
      right: 5px;
      background: var(--accent-green);
      color: white;
      border: 1px solid rgba(240, 246, 252, 0.1);
      padding: 0.45rem 0.95rem;
      border-radius: 5px;
      font-weight: 600;
      font-size: 0.82rem;
      cursor: pointer;
      transition: background 0.15s;
    }
    .submit-btn:hover {
      background: var(--accent-green-hover);
    }

    .results-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 1.5rem;
      display: none;
      flex-direction: column;
      gap: 1.15rem;
      animation: fadeIn 0.2s ease-in-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }

    .results-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 0.75rem;
    }
    .latency-badge {
      font-size: 0.72rem;
      font-family: 'JetBrains Mono', monospace;
      color: #3fb950;
      background: rgba(46, 160, 67, 0.1);
      border: 1px solid rgba(46, 160, 67, 0.25);
      padding: 0.2rem 0.55rem;
      border-radius: 4px;
    }
    .answer-box {
      font-size: 0.92rem;
      line-height: 1.68;
      color: #e6edf3;
    }
    .answer-box p {
      margin-bottom: 0.85rem;
    }
    .answer-box p:last-child {
      margin-bottom: 0;
    }
    .answer-box strong {
      color: #ffffff;
      font-weight: 600;
    }
    .answer-box em {
      color: #d2a8ff;
      font-style: italic;
    }
    .answer-box ul, .answer-box ol {
      margin: 0.5rem 0 0.85rem 1.4rem;
      padding-left: 0.2rem;
    }
    .answer-box li {
      margin-bottom: 0.35rem;
      line-height: 1.6;
    }
    .answer-box li::marker {
      color: #58a6ff;
    }
    .answer-box h1, .answer-box h2, .answer-box h3, .answer-box h4 {
      color: #f0f6fc;
      margin: 1rem 0 0.45rem 0;
      font-weight: 600;
    }
    .answer-box h1 { font-size: 1.15rem; border-bottom: 1px solid var(--border-subtle); padding-bottom: 0.3rem; }
    .answer-box h2 { font-size: 1.05rem; border-bottom: 1px solid var(--border-subtle); padding-bottom: 0.3rem; }
    .answer-box h3 { font-size: 0.98rem; }
    .answer-box h4 { font-size: 0.92rem; }
    .answer-box code {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.82rem;
      background: rgba(110, 118, 129, 0.2);
      color: #79c0ff;
      padding: 0.15rem 0.35rem;
      border-radius: 4px;
      border: 1px solid rgba(110, 118, 129, 0.25);
    }
    .answer-box pre {
      background: #0d1117;
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 0.85rem 1rem;
      overflow-x: auto;
      margin: 0.75rem 0;
    }
    .answer-box pre code {
      background: transparent;
      padding: 0;
      border: none;
      color: #e6edf3;
      font-size: 0.82rem;
    }
    .answer-box blockquote {
      border-left: 3px solid #58a6ff;
      margin: 0.75rem 0;
      padding-left: 0.85rem;
      color: var(--text-secondary);
    }

    .citations-header {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      border-top: 1px solid var(--border-subtle);
      padding-top: 1rem;
      margin-bottom: 0.65rem;
    }
    .citations-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 0.75rem;
    }
    .citation-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 0.85rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .citation-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .citation-name {
      font-weight: 600;
      font-size: 0.78rem;
      color: #58a6ff;
    }
    .citation-score {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.7rem;
      background: rgba(56, 139, 253, 0.1);
      color: #58a6ff;
      padding: 0.1rem 0.35rem;
      border-radius: 3px;
    }
    .citation-text {
      font-size: 0.75rem;
      line-height: 1.45;
      color: var(--text-secondary);
    }

    footer {
      background: var(--bg-surface);
      border-top: 1px solid var(--border-subtle);
      padding: 0.85rem 2rem;
      font-size: 0.75rem;
      color: var(--text-secondary);
      margin-top: auto;
    }
    .footer-inner {
      max-width: 1400px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    @media (max-width: 600px) {
      .footer-inner { flex-direction: column; gap: 0.4rem; align-items: flex-start; }
    }
    .footer-left {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }
    .footer-right {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }
    .footer-brand {
      font-weight: 600;
      color: var(--text-primary);
    }
    .footer-sep {
      color: var(--border-subtle);
    }
    .footer-link {
      color: var(--text-secondary);
      text-decoration: none;
      transition: color 0.15s;
    }
    .footer-link:hover {
      color: #58a6ff;
    }

    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 200;
    }
    .modal-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      width: 100%;
      max-width: 460px;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
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
      font-size: 1.35rem;
      cursor: pointer;
    }
  </style>
</head>
<body>

  <header>
    <a href="/" class="brand">
      <span class="brand-title">Neural RAG Engine</span>
      <span class="brand-sub">Dense Retrieval & LLM Grounding</span>
    </a>
    <div class="nav-actions">
      <button class="btn-outline" onclick="openSettings()">
        Settings
      </button>
      <a href="https://github.com/ozodbek-bosimov/neural-rag-engine" target="_blank" class="btn-github" title="View Source on GitHub">
        <svg height="16" width="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
        <span>GitHub</span>
      </a>
    </div>
  </header>

  <main class="container">
    <aside class="sidebar">
      <div class="section-title">
        <span>Knowledge Base</span>
        <div style="display:flex; align-items:center; gap:0.4rem;">
          <span id="doc-badge" class="chunk-tag">0 Docs</span>
          <button type="button" onclick="location.reload()" class="reset-btn" title="Refresh / Start new clean session">↺</button>
        </div>
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
          <span>Upload & Ingest</span>
        </div>

        <div class="ingest-tabs">
          <button type="button" id="tab-btn-file" class="ingest-tab active" onclick="switchIngestTab('file')">Upload File</button>
          <button type="button" id="tab-btn-text" class="ingest-tab" onclick="switchIngestTab('text')">Direct Text</button>
        </div>

        <div id="ingest-file-view" class="ingest-panel">
          <div class="upload-dropzone" id="upload-dropzone" onclick="document.getElementById('file-upload-input').click()">
            <svg width="24" height="24" viewBox="0 0 16 16" fill="currentColor" style="color:var(--text-secondary); margin-bottom:2px;"><path d="M7.47 10.78a.75.75 0 0 0 1.06 0l3.75-3.75a.75.75 0 0 0-1.06-1.06L8.75 8.44V1.75a.75.75 0 0 0-1.5 0v6.69L4.78 5.97a.75.75 0 0 0-1.06 1.06l3.75 3.75ZM3.75 13a.75.75 0 0 0 0 1.5h8.5a.75.75 0 0 0 0-1.5h-8.5Z"/></svg>
            <div class="upload-title">Choose file or drag & drop</div>
            <div class="upload-subtitle">PDF, TXT, MD, Code files (up to 25MB)</div>
            <input type="file" id="file-upload-input" accept=".pdf,.txt,.md,.json,.csv,.py" style="display:none" onchange="handleFileSelect(event)">
          </div>
        </div>

        <div id="ingest-text-view" class="ingest-panel" style="display:none;">
          <input type="text" id="ingest-title" class="input-box" placeholder="Document title...">
          <textarea id="ingest-content" class="input-box" placeholder="Paste document content or markdown..."></textarea>
          <button class="btn-primary" onclick="ingestDocument()">
            Add to Index
          </button>
        </div>

        <div id="upload-status" class="upload-status"></div>
      </div>
    </aside>

    <section class="main-view">
      <div class="search-panel">
        <div class="search-row">
          <input 
            type="text" 
            id="query-input" 
            class="search-bar" 
            placeholder="Ask a technical question across the indexed corpus..."
            onkeydown="if(event.key === 'Enter') submitQuery()"
          >
          <button id="search-btn" class="submit-btn" onclick="submitQuery()">
            Search
          </button>
        </div>
        <div class="demo-queries">
          <button class="demo-btn" onclick="applyQuery('What are the five essential steps of a standard PyTorch training loop?')">PyTorch Training Loop</button>
          <button class="demo-btn" onclick="applyQuery('What is RAG and how does it prevent LLM hallucination?')">RAG & Hallucination Grounding</button>
          <button class="demo-btn" onclick="applyQuery('What are the benefits of weight quantization and SIMD inference for deployment?')">SIMD & Quantization</button>
        </div>
      </div>

      <div id="results-card" class="results-card">
        <div class="results-top">
          <div style="display:flex; align-items:center; gap:0.6rem;">
            <span style="font-weight:600; font-size:0.85rem;">Answer:</span>
            <span id="model-name" style="font-size:0.75rem; color:var(--text-secondary); font-family:'JetBrains Mono', monospace;"></span>
          </div>
          <div id="latency-badge" class="latency-badge"></div>
        </div>

        <div id="answer-body" class="answer-box"></div>

        <div>
          <div class="citations-header">Retrieved Context Sources</div>
          <div id="citations-grid" class="citations-grid"></div>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <div class="footer-inner" style="justify-content:center;">
      <a href="https://github.com/ozodbek-bosimov/neural-rag-engine" target="_blank" class="footer-link" style="font-family:'JetBrains Mono', monospace; font-size:0.8rem;">https://github.com/ozodbek-bosimov/neural-rag-engine</a>
    </div>
  </footer>

  <div id="settings-modal" class="modal-backdrop">
    <div class="modal-card">
      <div class="modal-head">
        <h3 style="font-size:1rem; font-weight:600;">Engine & Model Settings</h3>
        <button class="modal-close-btn" onclick="closeSettings()">&times;</button>
      </div>
      <p style="font-size:0.8rem; color:var(--text-secondary); line-height:1.45;">
        Configure Google Gemini or OpenRouter API credentials to enable online LLM generation with citation grounding.
      </p>
      <div>
        <label style="display:block; font-size:0.75rem; font-weight:600; margin-bottom:0.35rem; color:var(--text-secondary);">API Key (Google Gemini or OpenRouter):</label>
        <input type="password" id="api-key-input" class="input-box" placeholder="Optional — leave empty to use default">
      </div>
      <div>
        <label style="display:block; font-size:0.75rem; font-weight:600; margin-bottom:0.35rem; color:var(--text-secondary);">Target Model:</label>
        <select id="model-select" class="input-box">
          <option value="gemini-flash-lite-latest">Google Gemini Flash-Lite (Fastest, Recommended)</option>
          <option value="gemini-flash-latest">Google Gemini Flash (Flagship)</option>
          <option value="deepseek/deepseek-r1:free">DeepSeek-R1 via OpenRouter</option>
          <option value="meta-llama/llama-3.3-70b-instruct:free">Meta Llama 3.3 70B via OpenRouter</option>
        </select>
      </div>
      <button class="btn-primary" onclick="saveSettings()">
        Save Settings
      </button>
    </div>
  </div>

  <script>
    // Isolated in-memory session per page load or browser refresh
    const SESSION_ID = 'sess_' + Math.random().toString(36).substring(2, 9) + Date.now().toString(36);
    let DEFAULT_GEMINI_KEY = "";
    let DEFAULT_MODEL = "gemini-flash-lite-latest";

    async function initAppConfig() {
      try {
        const res = await apiFetch('/api/config');
        if (res.ok) {
          const cfg = await res.json();
          if (cfg.default_gemini_key) DEFAULT_GEMINI_KEY = cfg.default_gemini_key;
          if (cfg.default_model) DEFAULT_MODEL = cfg.default_model;
        }
      } catch (e) {
        console.warn('Config load notice:', e);
      }
    }

    async function callGeminiDirect(query, sources, apiKey, model) {
      let targetModel = model || DEFAULT_MODEL;
      if (!targetModel.startsWith('gemini')) {
        targetModel = DEFAULT_MODEL;
      }
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${targetModel}:generateContent?key=${encodeURIComponent(apiKey)}`;
      const contextText = sources.map((s, idx) => `[Source ${idx+1}: ${s.document_title}]\n${s.text}`).join('\n\n');
      const systemInstruction = "You are an expert AI Engineer. Answer the user query strictly using the verified context sources provided below. Cite relevant source numbers where applicable. Use clean, well-formatted markdown without giant titles.";
      const prompt = `Context Sources:\n${contextText}\n\nUser Question:\n${query}\n\nProvide a concise, technically rigorous answer:`;

      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: `${systemInstruction}\n\n${prompt}` }] }],
          generationConfig: {
            temperature: 0.2,
            maxOutputTokens: 1024
          }
        })
      });

      if (!resp.ok) {
        throw new Error(`Gemini HTTP error ${resp.status}`);
      }
      const resJson = await resp.json();
      return resJson.candidates?.[0]?.content?.parts?.[0]?.text || '';
    }

    function apiFetch(url, options = {}) {
      options.headers = options.headers || {};
      options.headers['X-Session-ID'] = SESSION_ID;
      if (!options.headers['Content-Type'] && options.body && typeof options.body === 'string') {
        options.headers['Content-Type'] = 'application/json';
      }
      const separator = url.includes('?') ? '&' : '?';
      return fetch(url + separator + 'session_id=' + encodeURIComponent(SESSION_ID), options);
    }

    function renderMarkdown(md) {
      if (!md) return '';
      let cleanMd = md
        .replace(/^#+\s*(.{70,})$/gm, '$1')
        .replace(/([^\n])\s*##+\s*(\d*\.?\s*)/g, '$1\n\n**$2')
        .replace(/([^\n])\s*#+\s*/g, '$1\n\n');

      if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
        try {
          return marked.parse(cleanMd);
        } catch (e) {}
      }

      const text = cleanMd.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

      function parseInline(str) {
        let s = str
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');

        s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
        s = s.replace(/\*\*\*([^\*]+)\*\*\*/g, '<strong><em>$1</em></strong>');
        s = s.replace(/\*\*([^\*]+)\*\*/g, '<strong>$1</strong>');
        s = s.replace(/__([^_]+)__/g, '<strong>$1</strong>');
        s = s.replace(/(^|[^\*])\*([^\*\s][^\*]*[^\*\s]|[^\*\s])\*([^\*]|$)/g, '$1<em>$2</em>$3');
        s = s.replace(/(^|[^_])_([^_\s][^_]*[^_\s]|[^_\s])_([^_]|$)/g, '$1<em>$2</em>$3');
        return s;
      }

      const lines = text.split('\n');
      const output = [];
      let inList = false;
      let listType = 'ul';
      let inCodeBlock = false;
      let codeBuffer = [];

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        if (line.trim().startsWith('```')) {
          if (inCodeBlock) {
            inCodeBlock = false;
            const codeText = codeBuffer.join('\n')
              .replace(/&/g, '&amp;')
              .replace(/</g, '&lt;')
              .replace(/>/g, '&gt;');
            output.push(`<pre><code>${codeText}</code></pre>`);
            codeBuffer = [];
          } else {
            if (inList) {
              output.push(`</${listType}>`);
              inList = false;
            }
            inCodeBlock = true;
          }
          continue;
        }

        if (inCodeBlock) {
          codeBuffer.push(line);
          continue;
        }

        if (!line.trim()) {
          if (inList) {
            output.push(`</${listType}>`);
            inList = false;
          }
          continue;
        }

        const ulMatch = line.match(/^\s*[\*\-]\s+(.*)$/);
        if (ulMatch) {
          if (!inList || listType !== 'ul') {
            if (inList) output.push(`</${listType}>`);
            output.push('<ul>');
            inList = true;
            listType = 'ul';
          }
          output.push(`<li>${parseInline(ulMatch[1])}</li>`);
          continue;
        }

        const olMatch = line.match(/^\s*(\d+)\.\s+(.*)$/);
        if (olMatch) {
          if (!inList || listType !== 'ol') {
            if (inList) output.push(`</${listType}>`);
            output.push('<ol>');
            inList = true;
            listType = 'ol';
          }
          output.push(`<li>${parseInline(olMatch[2])}</li>`);
          continue;
        }

        if (inList) {
          output.push(`</${listType}>`);
          inList = false;
        }

        if (line.startsWith('### ')) {
          output.push(`<h4>${parseInline(line.substring(4))}</h4>`);
          continue;
        }
        if (line.startsWith('## ')) {
          output.push(`<h3>${parseInline(line.substring(3))}</h3>`);
          continue;
        }
        if (line.startsWith('# ')) {
          output.push(`<h2>${parseInline(line.substring(2))}</h2>`);
          continue;
        }

        if (line.startsWith('> ')) {
          output.push(`<blockquote>${parseInline(line.substring(2))}</blockquote>`);
          continue;
        }

        output.push(`<p>${parseInline(line)}</p>`);
      }

      if (inList) {
        output.push(`</${listType}>`);
      }
      if (inCodeBlock) {
        const codeText = codeBuffer.join('\n')
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');
        output.push(`<pre><code>${codeText}</code></pre>`);
      }

      return output.join('\n');
    }

    async function refreshStats() {
      try {
        const res = await apiFetch('/api/stats');
        const data = await res.json();
        document.getElementById('stat-chunks').textContent = data.total_chunks;
        document.getElementById('stat-dim').textContent = data.embedding_dimension;
        document.getElementById('doc-badge').textContent = data.total_documents + ' Docs';

        const listEl = document.getElementById('doc-list');
        listEl.innerHTML = '';
        data.indexed_documents.forEach(doc => {
          const div = document.createElement('div');
          div.className = 'doc-item';
          const docId = doc.document_id || doc.title || '';
          const title = doc.title || doc.document_id || '';
          const count = doc.chunk_count || 0;
          const chunkLabel = count === 1 ? '1 chunk' : `${count} chunks`;
          div.innerHTML = `
            <span class="doc-title" title="${title}">
              <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor" style="color:var(--text-secondary); flex-shrink:0; margin-right:6px;"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.793V4.25c0 .138.112.25.25.25h2.457Z"/></svg>
              <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${title}</span>
            </span>
            <div style="display:flex; align-items:center; gap:0.4rem; flex-shrink:0;">
              <span class="chunk-tag">${chunkLabel}</span>
              <button class="delete-btn" onclick="deleteDocument('${encodeURIComponent(docId)}', '${encodeURIComponent(title)}')" title="Delete from knowledge index">✕</button>
            </div>
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
        const apiKey = localStorage.getItem('api_key') || DEFAULT_GEMINI_KEY;
        const savedModel = localStorage.getItem('rag_model') || DEFAULT_MODEL;

        const res = await apiFetch('/api/query', {
          method: 'POST',
          body: JSON.stringify({ query: query, model: savedModel, api_key: apiKey })
        });
        const data = await res.json();

        let finalAnswer = data.answer;
        let finalModel = data.model_used;
        let finalFallback = data.fallback_used;

        // If server had to use offline fallback (e.g. Alwaysdata server IP blocked by Google GFE),
        // invoke Google Gemini directly from the visitor browser using the active Gemini key!
        if (data.fallback_used && apiKey && (apiKey.startsWith('AQ.') || apiKey.startsWith('AIza')) && data.sources && data.sources.length > 0) {
          try {
            const directAnswer = await callGeminiDirect(query, data.sources, apiKey, savedModel);
            if (directAnswer && directAnswer.trim()) {
              finalAnswer = directAnswer;
              finalModel = `Google Gemini (${savedModel.replace('models/', '')})`;
              finalFallback = false;
            }
          } catch (geminiErr) {
            console.warn('Direct Gemini call fallback notice:', geminiErr);
          }
        }

        card.style.display = 'flex';
        answerEl.innerHTML = renderMarkdown(finalAnswer);
        modelEl.textContent = finalModel + (finalFallback ? ' (Offline Mode)' : '');
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

    function switchIngestTab(tab) {
      const fileTab = document.getElementById('tab-btn-file');
      const textTab = document.getElementById('tab-btn-text');
      const fileView = document.getElementById('ingest-file-view');
      const textView = document.getElementById('ingest-text-view');

      if (tab === 'file') {
        fileTab.classList.add('active');
        textTab.classList.remove('active');
        fileView.style.display = 'block';
        textView.style.display = 'none';
      } else {
        textTab.classList.add('active');
        fileTab.classList.remove('active');
        fileView.style.display = 'none';
        textView.style.display = 'flex';
      }
    }

    async function ingestDocument() {
      const title = document.getElementById('ingest-title').value.trim() || 'Custom Document';
      const content = document.getElementById('ingest-content').value.trim();
      if (!content) {
        setUploadStatus('Please enter document content before indexing.', 'error');
        return;
      }

      setUploadStatus(`Indexing ${title}...`, 'loading');
      try {
        const res = await apiFetch('/api/ingest', {
          method: 'POST',
          body: JSON.stringify({ title: title, content: content })
        });
        const data = await res.json();
        if (res.ok) {
          setUploadStatus(`✓ Indexed "${title}" (${data.chunks_created} chunks added)`, 'success');
          document.getElementById('ingest-title').value = '';
          document.getElementById('ingest-content').value = '';
          refreshStats();
          setTimeout(() => setUploadStatus('', ''), 4000);
        } else {
          setUploadStatus(`Error: ${data.error || 'Failed to index'}`, 'error');
        }
      } catch (err) {
        setUploadStatus(`Ingestion error: ${err}`, 'error');
      }
    }

    function openSettings() {
      document.getElementById('settings-modal').style.display = 'flex';
      document.getElementById('api-key-input').value = localStorage.getItem('api_key') || '';
      document.getElementById('model-select').value = localStorage.getItem('rag_model') || DEFAULT_MODEL;
    }

    function closeSettings() {
      document.getElementById('settings-modal').style.display = 'none';
    }

    async function saveSettings() {
      const key = document.getElementById('api-key-input').value.trim();
      const model = document.getElementById('model-select').value;
      if (key) localStorage.setItem('api_key', key);
      localStorage.setItem('rag_model', model);

      await apiFetch('/api/settings', {
        method: 'POST',
        body: JSON.stringify({ api_key: key, model: model })
      });
      closeSettings();
      alert('Settings saved successfully.');
    }

    function setUploadStatus(message, type) {
      const el = document.getElementById('upload-status');
      if (!message) {
        el.style.display = 'none';
        el.className = 'upload-status';
        el.textContent = '';
        return;
      }
      el.textContent = message;
      el.className = `upload-status ${type}`;
    }

    function setupDragAndDrop() {
      const dropzone = document.getElementById('upload-dropzone');
      if (!dropzone) return;

      ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
          e.preventDefault();
          e.stopPropagation();
          dropzone.classList.add('dragover');
        });
      });

      ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
          e.preventDefault();
          e.stopPropagation();
          dropzone.classList.remove('dragover');
        });
      });

      dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
          uploadFile(files[0]);
        }
      });
    }

    function handleFileSelect(event) {
      const file = event.target.files[0];
      if (file) {
        uploadFile(file);
      }
    }

    function uploadFile(file) {
      setUploadStatus(`Indexing ${file.name}...`, 'loading');
      const isPdf = file.name.toLowerCase().endsWith('.pdf');
      const reader = new FileReader();

      if (isPdf) {
        reader.onload = async function(e) {
          const b64 = e.target.result;
          await sendFilePayload(file.name, { base64: b64 });
        };
        reader.readAsDataURL(file);
      } else {
        reader.onload = async function(e) {
          const txt = e.target.result;
          await sendFilePayload(file.name, { text: txt });
        };
        reader.readAsText(file);
      }
    }

    async function sendFilePayload(filename, payloadData) {
      try {
        const res = await apiFetch('/api/upload', {
          method: 'POST',
          body: JSON.stringify({ filename: filename, ...payloadData })
        });
        const data = await res.json();
        if (res.ok) {
          setUploadStatus(`✓ Indexed ${filename} (${data.chunks_created} chunks added)`, 'success');
          refreshStats();
          setTimeout(() => setUploadStatus('', ''), 4500);
        } else {
          setUploadStatus(`Error: ${data.error || 'Failed to index file'}`, 'error');
        }
      } catch (err) {
        setUploadStatus(`Upload error: ${err}`, 'error');
      }
    }

    async function deleteDocument(encodedId, encodedTitle) {
      const docId = decodeURIComponent(encodedId);
      const title = decodeURIComponent(encodedTitle);
      setUploadStatus(`Deleting ${title}...`, 'loading');
      try {
        const res = await apiFetch('/api/delete', {
          method: 'POST',
          body: JSON.stringify({ document_id: docId })
        });
        const data = await res.json();
        if (res.ok) {
          setUploadStatus(`✓ Deleted "${title}" (${data.deleted_chunks} chunks removed)`, 'success');
          refreshStats();
          setTimeout(() => setUploadStatus('', ''), 3500);
        } else {
          setUploadStatus(`Error: ${data.error || 'Failed to delete document'}`, 'error');
        }
      } catch (err) {
        setUploadStatus(`Delete error: ${err}`, 'error');
      }
    }

    window.addEventListener('DOMContentLoaded', () => {
      initAppConfig();
      refreshStats();
      setupDragAndDrop();
    });
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
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Session-ID")
        self.end_headers()

    def _get_session_id(self, body_data: Optional[Dict] = None) -> str:
        sid = self.headers.get("X-Session-ID")
        if sid and sid.strip():
            return sid.strip()
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if "session_id" in qs and qs["session_id"]:
            return qs["session_id"][0].strip()
        if body_data and isinstance(body_data, dict):
            bsid = body_data.get("session_id")
            if bsid and str(bsid).strip():
                return str(bsid).strip()
        return "default"

    def _get_engine(self, body_data: Optional[Dict] = None) -> RAGEngine:
        sid = self._get_session_id(body_data)
        return get_engine_for_session(sid)

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._set_headers(200, "text/html; charset=utf-8")
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if path == "/api/config":
            cfg = {
                "default_gemini_key": GLOBAL_API_KEY or os.getenv("GEMINI_API_KEY", ""),
                "default_model": GLOBAL_MODEL or os.getenv("DEFAULT_MODEL", "gemini-flash-lite-latest")
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(cfg).encode("utf-8"))
            return

        if path == "/api/stats":
            engine = self._get_engine()
            stats = engine.get_system_stats()
            self._set_headers(200)
            self.wfile.write(json.dumps(stats).encode("utf-8"))
            return

        if path == "/api/health":
            engine = self._get_engine()
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "healthy",
                "indexed_documents": len(engine.indexed_docs),
                "indexed_chunks": engine.vector_store.count(),
                "active_sessions": len(sessions)
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

        engine = self._get_engine(body)

        if path == "/api/query":
            query_text = body.get("query", "").strip()
            custom_model = body.get("model", None)
            custom_api_key = body.get("api_key", None)
            top_k = int(body.get("top_k", 3))

            if not query_text:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Empty query"}).encode("utf-8"))
                return

            res = engine.query(query_text, top_k=top_k, custom_model=custom_model, custom_api_key=custom_api_key)
            self._set_headers(200)
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        if path == "/api/upload":
            raw_filename = body.get("filename", "uploaded_doc").strip()
            filename = os.path.basename(raw_filename)
            b64_data = body.get("base64")
            text_data = body.get("text")

            if b64_data:
                try:
                    if "," in b64_data:
                        b64_data = b64_data.split(",", 1)[1]
                    file_bytes = base64.b64decode(b64_data)

                    if filename.lower().endswith(".pdf"):
                        chunks_cnt = engine.ingest_pdf(filename, file_bytes)
                    else:
                        text = file_bytes.decode("utf-8", errors="ignore")
                        title = filename.rsplit(".", 1)[0].replace("_", " ").title()
                        chunks_cnt = engine.ingest_text(filename, title, text)
                except Exception as e:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": f"Failed to process file: {e}"}).encode("utf-8"))
                    return
            elif text_data:
                try:
                    title = filename.rsplit(".", 1)[0].replace("_", " ").title()
                    chunks_cnt = engine.ingest_text(filename, title, text_data)
                except Exception as e:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": f"Failed to process text: {e}"}).encode("utf-8"))
                    return
            else:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "No file content provided"}).encode("utf-8"))
                return

            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "success",
                "filename": filename,
                "chunks_created": chunks_cnt,
                "total_chunks": engine.vector_store.count()
            }).encode("utf-8"))
            return

        if path == "/api/delete":
            document_id = body.get("document_id", "").strip()
            if not document_id:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "document_id is required"}).encode("utf-8"))
                return

            deleted_chunks = engine.delete_document(document_id)

            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "success",
                "document_id": document_id,
                "deleted_chunks": deleted_chunks,
                "total_documents": len(engine.indexed_docs),
                "total_chunks": engine.vector_store.count()
            }).encode("utf-8"))
            return

        if path == "/api/ingest":
            title = body.get("title", "Document").strip()
            content = body.get("content", "").strip()
            doc_id = title.lower().replace(" ", "_") + ".txt"
            safe_doc_id = os.path.basename(doc_id)

            if not content:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Document content is empty"}).encode("utf-8"))
                return

            chunks_cnt = engine.ingest_text(safe_doc_id, title, content)
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "success",
                "document_id": safe_doc_id,
                "chunks_created": chunks_cnt
            }).encode("utf-8"))
            return

        if path == "/api/settings":
            api_key = body.get("api_key", "").strip()
            model = body.get("model", "").strip()
            global GLOBAL_API_KEY, GLOBAL_MODEL
            if api_key:
                GLOBAL_API_KEY = api_key
                if api_key.startswith("AQ.") or "AIza" in api_key:
                    engine.llm_client.gemini_key = api_key
                    engine.llm_client.provider = "Google Gemini"
                elif api_key.startswith("sk-"):
                    engine.llm_client.openrouter_key = api_key
                    engine.llm_client.provider = "OpenRouter"
                engine.llm_client.api_key = api_key
            if model:
                GLOBAL_MODEL = model
                engine.llm_client.model = model
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "settings_updated",
                "provider": engine.llm_client.provider,
                "model": engine.llm_client.model
            }).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Not Found"}).encode("utf-8"))


def run_server(port: int = 8085):
    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, RAGRequestHandler)
    httpd.daemon_threads = True
    print(f"[*] Neural RAG Engine Server running on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Server stopped.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8085
    run_server(port)
