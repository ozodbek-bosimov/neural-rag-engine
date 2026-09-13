import re
from typing import List, Dict, Any


class DocumentChunker:
    """Sentence-preserving sliding window text chunker with configurable overlap."""

    def __init__(self, chunk_size: int = 120, chunk_overlap: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_sentences(self, text: str) -> List[str]:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, text: str, document_id: str = "doc") -> List[Dict[str, Any]]:
        sentences = self.split_sentences(text)
        chunks = []
        current = []
        current_words = 0

        for sentence in sentences:
            words = len(sentence.split())
            if current_words + words > self.chunk_size and current:
                chunk_text = " ".join(current)
                chunks.append({
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_c{len(chunks)}",
                    "text": chunk_text,
                    "word_count": len(chunk_text.split())
                })

                overlap_chunk = []
                overlap_words = 0
                for s in reversed(current):
                    w = len(s.split())
                    if overlap_words + w <= self.chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_words += w
                    else:
                        break

                current = overlap_chunk
                current_words = overlap_words

            current.append(sentence)
            current_words += words

        if current:
            chunk_text = " ".join(current)
            chunks.append({
                "document_id": document_id,
                "chunk_id": f"{document_id}_c{len(chunks)}",
                "text": chunk_text,
                "word_count": len(chunk_text.split())
            })

        return chunks
