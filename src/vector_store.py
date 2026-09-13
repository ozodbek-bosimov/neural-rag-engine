from typing import List, Dict, Any, Tuple


class VectorStore:
    """In-memory dense vector store with cosine similarity ranking."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: List[List[float]] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(self, embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]):
        if not embeddings:
            return
        self.vectors.extend(embeddings)
        self.metadata.extend(metadata_list)

    def search(self, query_vector: List[float], top_k: int = 3, min_score: float = 0.0) -> List[Dict[str, Any]]:
        if not self.vectors:
            return []

        scores: List[Tuple[int, float]] = []
        for idx, vec in enumerate(self.vectors):
            dot_product = sum(a * b for a, b in zip(vec, query_vector))
            scores.append((idx, dot_product))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            if score >= min_score:
                item = self.metadata[idx].copy()
                item["similarity_score"] = round(float(score), 4)
                results.append(item)

        return results

    def count(self) -> int:
        return len(self.metadata)

    def delete_by_document_id(self, document_id: str) -> int:
        if not document_id or not document_id.strip() or not self.metadata:
            return 0
        new_vectors = []
        new_metadata = []
        deleted_count = 0
        doc_lower = document_id.strip().lower()
        for vec, meta in zip(self.vectors, self.metadata):
            m_id = str(meta.get("document_id", "")).lower()
            m_title = str(meta.get("document_title", "")).lower()
            if m_id == doc_lower or m_title == doc_lower:
                deleted_count += 1
            else:
                new_vectors.append(vec)
                new_metadata.append(meta)
        self.vectors = new_vectors
        self.metadata = new_metadata
        return deleted_count

    def clear(self):
        self.vectors.clear()
        self.metadata.clear()

