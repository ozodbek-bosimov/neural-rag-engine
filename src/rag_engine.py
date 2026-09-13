import os
import re
import glob
import time
import math
import hashlib
from typing import List, Dict, Any, Optional

try:
    from src.chunker import DocumentChunker
    from src.vector_store import VectorStore
    from src.model import EmbeddingProjectionHead
    from src.llm_client import LLMClient
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from chunker import DocumentChunker
    from vector_store import VectorStore
    from model import EmbeddingProjectionHead
    from llm_client import LLMClient


class RAGEngine:
    """End-to-end RAG pipeline: chunking, dense embedding projection, retrieval, and synthesis."""

    def __init__(
        self,
        embedding_dim: int = 384,
        chunk_size: int = 120,
        chunk_overlap: int = 30,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.embedding_dim = embedding_dim
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore(dimension=embedding_dim)
        self.projection_head = EmbeddingProjectionHead(input_dim=embedding_dim, hidden_dim=256, output_dim=256)
        self.llm_client = LLMClient(api_key=api_key, model=model_name)
        self.indexed_docs: Dict[str, Dict[str, Any]] = {}

    STOP_WORDS = {
        'what', 'is', 'and', 'how', 'does', 'it', 'the', 'in', 'a', 'an',
        'of', 'to', 'for', 'with', 'are', 'this', 'that', 'from', 'by', 'as'
    }

    @staticmethod
    def _stem(word: str) -> str:
        for suffix in ('ing', 'tion', 'tions', 'ed', 'es', 's'):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

    def compute_dense_embedding(self, text: str) -> List[float]:
        raw_words = re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
        if not raw_words:
            return [0.0] * self.embedding_dim

        vec = [0.0] * self.embedding_dim
        for i, raw in enumerate(raw_words):
            if raw in self.STOP_WORDS:
                continue

            stem = self._stem(raw)

            # Exact word token hash
            h1 = int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16) % self.embedding_dim
            # Stem token hash
            h_stem = int(hashlib.sha256(stem.encode("utf-8")).hexdigest(), 16) % self.embedding_dim
            # Subword prefix hash
            h_sub = int(hashlib.md5(stem[:4].encode("utf-8")).hexdigest(), 16) % self.embedding_dim

            vec[h1] += 2.0
            vec[h_stem] += 2.5
            vec[h_sub] += 1.0

            if i > 0 and raw_words[i-1] not in self.STOP_WORDS:
                bi = f"{self._stem(raw_words[i-1])}_{stem}"
                hb = int(hashlib.sha256(bi.encode("utf-8")).hexdigest(), 16) % self.embedding_dim
                vec[hb] += 2.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]

        return vec

    def ingest_text(self, document_id: str, title: str, content: str) -> int:
        chunks = self.chunker.chunk_document(content, document_id=document_id)
        if not chunks:
            return 0

        embeddings = []
        metadata_list = []
        for chunk in chunks:
            emb = self.compute_dense_embedding(chunk["text"])
            embeddings.append(emb)
            metadata_list.append({
                "document_id": document_id,
                "document_title": title,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "word_count": chunk["word_count"]
            })

        self.vector_store.add(embeddings, metadata_list)
        self.indexed_docs[document_id] = {
            "title": title,
            "chunk_count": len(chunks),
            "char_count": len(content)
        }
        return len(chunks)

    def ingest_file(self, file_path: str) -> int:
        if not os.path.exists(file_path):
            return 0

        doc_id = os.path.basename(file_path)
        title = doc_id.replace("_", " ").replace(".md", "").replace(".txt", "").title()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.ingest_text(doc_id, title, content)
        except Exception as e:
            print(f"[!] Error reading file {file_path}: {e}")
            return 0

    def ingest_directory(self, dir_path: str) -> Dict[str, int]:
        stats = {}
        if not os.path.isdir(dir_path):
            return stats

        files = sorted(glob.glob(os.path.join(dir_path, "*.md")) + glob.glob(os.path.join(dir_path, "*.txt")))
        for fp in files:
            doc_id = os.path.basename(fp)
            cnt = self.ingest_file(fp)
            stats[doc_id] = cnt
        return stats

    def query(
        self,
        user_query: str,
        top_k: int = 3,
        min_similarity: float = 0.05,
        custom_model: Optional[str] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        q_emb = self.compute_dense_embedding(user_query)
        retrieved_chunks = self.vector_store.search(q_emb, top_k=top_k, min_score=min_similarity)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not retrieved_chunks:
            return {
                "answer": "No relevant context found in indexed documents for this query.",
                "sources": [],
                "retrieval_time_ms": round(retrieval_ms, 2),
                "generation_time_ms": 0.0,
                "total_latency_ms": round(retrieval_ms, 2),
                "model_used": "no-context-fallback"
            }

        context_parts = []
        for i, c in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Source {i}: {c['document_title']} (Similarity: {c['similarity_score']*100:.1f}%)]\n{c['text']}"
            )
        full_context = "\n\n".join(context_parts)

        grounded_prompt = (
            "You are an expert AI Engineer assistant.\n"
            "Answer the user query strictly using the verified context below.\n"
            "Do not hallucinate or state facts not supported by the context.\n\n"
            f"=== RETRIEVED CONTEXT ===\n{full_context}\n\n"
            f"=== USER QUERY ===\n{user_query}\n\n"
            "Provide a concise, technically rigorous answer:"
        )

        t_gen_start = time.perf_counter()
        if custom_model:
            prev_model = self.llm_client.model
            self.llm_client.model = custom_model
            llm_response = self.llm_client.generate(prompt=grounded_prompt, system_prompt="You are a professional AI engineer.")
            self.llm_client.model = prev_model
        else:
            llm_response = self.llm_client.generate(prompt=grounded_prompt, system_prompt="You are a professional AI engineer.")

        generation_ms = (time.perf_counter() - t_gen_start) * 1000
        total_latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "query": user_query,
            "answer": llm_response.get("text", ""),
            "sources": retrieved_chunks,
            "retrieval_time_ms": round(retrieval_ms, 2),
            "generation_time_ms": round(generation_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "model_used": llm_response.get("model", "unknown"),
            "provider": llm_response.get("provider", "local"),
            "fallback_used": llm_response.get("is_fallback", False),
            "context_length": len(full_context)
        }

    def get_system_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.indexed_docs),
            "total_chunks": self.vector_store.count(),
            "embedding_dimension": self.embedding_dim,
            "indexed_documents": list(self.indexed_docs.values()),
            "llm_api_configured": self.llm_client.api_key is not None
        }
