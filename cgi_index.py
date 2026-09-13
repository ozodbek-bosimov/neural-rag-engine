#!/usr/bin/env python3
import os
import sys
import json
import time
import pickle
import base64
from urllib.parse import parse_qs, urlparse

# Ensure neural-rag-engine is in sys.path
APP_DIR = "/home/neural-rag/neural-rag-engine"
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Load .env
env_file = os.path.join(APP_DIR, ".env")
if os.path.isfile(env_file):
    try:
        with open(env_file, "r", encoding="utf-8") as f:
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

from src.rag_engine import RAGEngine
import app

SESSIONS_DIR = "/home/neural-rag/sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)
DATA_DIR = os.path.join(APP_DIR, "data", "sample_docs")
GLOBAL_API_KEY = os.getenv("GEMINI_API_KEY")
GLOBAL_MODEL = os.getenv("DEFAULT_MODEL", "gemini-flash-lite-latest")

def get_session_file(sid: str) -> str:
    clean_id = "".join(c for c in sid if c.isalnum() or c in ("-", "_"))[:64]
    if not clean_id:
        clean_id = "default"
    return os.path.join(SESSIONS_DIR, f"{clean_id}.pkl")

def cleanup_old_session_files():
    try:
        now = time.time()
        for fname in os.listdir(SESSIONS_DIR):
            if fname.endswith(".pkl") and fname != "default.pkl":
                p = os.path.join(SESSIONS_DIR, fname)
                if now - os.path.getmtime(p) > 86400:  # 24 hours
                    try:
                        os.remove(p)
                    except Exception:
                        pass
    except Exception:
        pass

def load_or_create_engine(sid: str) -> RAGEngine:
    cleanup_old_session_files()
    fpath = get_session_file(sid)
    if os.path.exists(fpath):
        try:
            with open(fpath, "rb") as f:
                eng = pickle.load(f)
                if GLOBAL_API_KEY and not eng.llm_client.api_key:
                    eng.llm_client.api_key = GLOBAL_API_KEY
                    eng.llm_client.gemini_key = GLOBAL_API_KEY
                return eng
        except Exception:
            pass
    # Create fresh engine with sample docs
    eng = RAGEngine(api_key=GLOBAL_API_KEY, model_name=GLOBAL_MODEL)
    if os.path.exists(DATA_DIR):
        eng.ingest_directory(DATA_DIR)
    save_engine(sid, eng)
    return eng

def save_engine(sid: str, eng: RAGEngine):
    fpath = get_session_file(sid)
    try:
        tmp_path = fpath + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(eng, f)
        os.replace(tmp_path, fpath)
    except Exception as e:
        sys.stderr.write(f"Failed to save session {sid}: {e}\n")

REAL_STDOUT = sys.stdout
REAL_STDOUT_BUFFER = sys.stdout.buffer
# Route all internal library/app prints to stderr so CGI stdout is clean
sys.stdout = sys.stderr

def respond(status_code: int, content_type: str, body_bytes: bytes, extra_headers=None, is_head=False):
    status_text = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}.get(status_code, "OK")
    REAL_STDOUT.write(f"Status: {status_code} {status_text}\r\n")
    REAL_STDOUT.write(f"Content-Type: {content_type}\r\n")
    REAL_STDOUT.write("Access-Control-Allow-Origin: *\r\n")
    REAL_STDOUT.write("Access-Control-Allow-Methods: GET, POST, HEAD, OPTIONS\r\n")
    REAL_STDOUT.write("Access-Control-Allow-Headers: Content-Type, X-Session-Id\r\n")
    if extra_headers:
        for k, v in extra_headers.items():
            REAL_STDOUT.write(f"{k}: {v}\r\n")
    REAL_STDOUT.write(f"Content-Length: {len(body_bytes)}\r\n\r\n")
    REAL_STDOUT.flush()
    if not is_head:
        REAL_STDOUT_BUFFER.write(body_bytes)
        REAL_STDOUT_BUFFER.flush()

def main():
    method = os.environ.get("REQUEST_METHOD", "GET").upper()
    req_uri = os.environ.get("REQUEST_URI", "/")
    parsed_url = urlparse(req_uri)
    path = parsed_url.path or "/"
    query_params = parse_qs(parsed_url.query or os.environ.get("QUERY_STRING", ""))
    is_head = (method == "HEAD")

    if method == "OPTIONS":
        respond(200, "text/plain", b"")
        return

    # Extract session ID from headers or query
    session_id = os.environ.get("HTTP_X_SESSION_ID", "").strip()
    if not session_id and "session_id" in query_params:
        session_id = query_params["session_id"][0].strip()

    if method in ("GET", "HEAD"):
        if path in ("/", "/index.html"):
            respond(200, "text/html; charset=utf-8", app.HTML_PAGE.encode("utf-8"), is_head=is_head)
            return

        if path == "/api/stats":
            engine = load_or_create_engine(session_id)
            stats = engine.get_system_stats()
            respond(200, "application/json", json.dumps(stats).encode("utf-8"), is_head=is_head)
            return

        if path == "/api/health":
            engine = load_or_create_engine(session_id)
            active_count = len([f for f in os.listdir(SESSIONS_DIR) if f.endswith(".pkl")])
            res = {
                "status": "healthy",
                "indexed_documents": len(engine.indexed_docs),
                "indexed_chunks": engine.vector_store.count(),
                "active_sessions": active_count
            }
            respond(200, "application/json", json.dumps(res).encode("utf-8"), is_head=is_head)
            return

        respond(404, "application/json", json.dumps({"error": "Not Found"}).encode("utf-8"), is_head=is_head)
        return

    if method == "POST":
        content_len = int(os.environ.get("CONTENT_LENGTH", 0))
        raw_body = sys.stdin.buffer.read(content_len) if content_len > 0 else b"{}"
        try:
            body = json.loads(raw_body.decode("utf-8", errors="ignore"))
        except Exception:
            body = {}

        if not session_id:
            session_id = str(body.get("session_id", "")).strip()

        engine = load_or_create_engine(session_id)

        if path == "/api/query":
            query_text = body.get("query", "").strip()
            custom_model = body.get("model", None)
            top_k = int(body.get("top_k", 3))

            if not query_text:
                respond(400, "application/json", json.dumps({"error": "Empty query"}).encode("utf-8"))
                return

            res = engine.query(query_text, top_k=top_k, custom_model=custom_model)
            respond(200, "application/json", json.dumps(res).encode("utf-8"))
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
                    save_engine(session_id, engine)
                except Exception as e:
                    respond(400, "application/json", json.dumps({"error": f"Failed to process file: {e}"}).encode("utf-8"))
                    return
            elif text_data:
                try:
                    title = filename.rsplit(".", 1)[0].replace("_", " ").title()
                    chunks_cnt = engine.ingest_text(filename, title, text_data)
                    save_engine(session_id, engine)
                except Exception as e:
                    respond(400, "application/json", json.dumps({"error": f"Failed to process text: {e}"}).encode("utf-8"))
                    return
            else:
                respond(400, "application/json", json.dumps({"error": "No file content provided"}).encode("utf-8"))
                return

            respond(200, "application/json", json.dumps({
                "status": "success",
                "filename": filename,
                "chunks_created": chunks_cnt,
                "total_chunks": engine.vector_store.count()
            }).encode("utf-8"))
            return

        if path == "/api/delete":
            document_id = body.get("document_id", "").strip()
            if not document_id:
                respond(400, "application/json", json.dumps({"error": "document_id is required"}).encode("utf-8"))
                return
            deleted_chunks = engine.delete_document(document_id)
            save_engine(session_id, engine)
            respond(200, "application/json", json.dumps({
                "status": "success",
                "document_id": document_id,
                "deleted_chunks": deleted_chunks,
                "total_documents": len(engine.indexed_docs),
                "total_chunks": engine.vector_store.count()
            }).encode("utf-8"))
            return

        if path == "/api/ingest":
            title = body.get("title", "Document").strip()
            content = (body.get("content") or body.get("text") or "").strip()
            doc_id = title.lower().replace(" ", "_") + ".txt"
            safe_doc_id = os.path.basename(doc_id)
            if not content:
                respond(400, "application/json", json.dumps({"error": "Document content is empty"}).encode("utf-8"))
                return
            chunks_cnt = engine.ingest_text(safe_doc_id, title, content)
            save_engine(session_id, engine)
            respond(200, "application/json", json.dumps({
                "status": "success",
                "document_id": safe_doc_id,
                "chunks_created": chunks_cnt
            }).encode("utf-8"))
            return

        if path == "/api/settings":
            api_key = body.get("api_key", "").strip()
            model = body.get("model", "").strip()
            if api_key:
                if api_key.startswith("AQ.") or "AIza" in api_key:
                    engine.llm_client.gemini_key = api_key
                    engine.llm_client.provider = "Google Gemini"
                elif api_key.startswith("sk-"):
                    engine.llm_client.openrouter_key = api_key
                    engine.llm_client.provider = "OpenRouter"
                engine.llm_client.api_key = api_key
            if model:
                engine.llm_client.model = model
            save_engine(session_id, engine)
            respond(200, "application/json", json.dumps({
                "status": "settings_updated",
                "provider": engine.llm_client.provider,
                "model": engine.llm_client.model
            }).encode("utf-8"))
            return

        respond(404, "application/json", json.dumps({"error": "Not Found"}).encode("utf-8"))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        sys.stderr.write(f"Unhandled error in CGI: {err_msg}\n")
        try:
            respond(500, "application/json", json.dumps({"error": str(e), "traceback": err_msg}).encode("utf-8"))
        except Exception:
            pass
