import unittest
from src.dataset import SemanticPairDataset
from src.model import EmbeddingProjectionHead, HAS_TORCH
from src.chunker import DocumentChunker
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine


class TestPipeline(unittest.TestCase):

    def test_semantic_dataset(self):
        pairs = [
            ([0.1, 0.2], [0.15, 0.25], 1.0),
            ([0.9, -0.2], [-0.8, 0.3], -1.0)
        ]
        dataset = SemanticPairDataset(pairs)
        self.assertEqual(len(dataset), 2)
        x1, x2, y = dataset[0]
        if HAS_TORCH:
            self.assertEqual(tuple(x1.shape), (2,))
            self.assertEqual(float(y.item()), 1.0)
        else:
            self.assertEqual(len(x1), 2)
            self.assertEqual(y, 1.0)

    def test_projection_head(self):
        model = EmbeddingProjectionHead(input_dim=384, hidden_dim=256, output_dim=256)
        if HAS_TORCH:
            import torch
            sample_input = torch.randn(2, 384)
            output = model(sample_input)
            self.assertEqual(output.shape, (2, 256))
            norm = torch.norm(output, p=2, dim=-1)
            self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-4))
        else:
            sample_vec = [0.1] * 384
            res = model.forward(sample_vec)
            self.assertEqual(len(res), 384)

    def test_document_chunker(self):
        chunker = DocumentChunker(chunk_size=15, chunk_overlap=5)
        text = "PyTorch is an open-source deep learning framework. It provides accelerated tensor computations. Autograd computes automatic derivatives. Training loops optimize model weights."
        chunks = chunker.chunk_document(text, document_id="doc_test")
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertTrue(c["word_count"] > 0)
            self.assertEqual(c["document_id"], "doc_test")

    def test_vector_store(self):
        store = VectorStore(dimension=3)
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        v3 = [0.7071, 0.7071, 0.0]

        meta = [
            {"id": "doc_x", "text": "Vector X"},
            {"id": "doc_y", "text": "Vector Y"},
            {"id": "doc_xy", "text": "Vector XY"},
        ]
        store.add([v1, v2, v3], meta)

        results = store.search([1.0, 0.0, 0.0], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "doc_x")
        self.assertAlmostEqual(results[0]["similarity_score"], 1.0, places=2)

    def test_rag_engine(self):
        engine = RAGEngine()
        engine.ingest_text(
            document_id="faq.md",
            title="Interview Questions",
            content="PyTorch uses optimizer.zero_grad() to clear past gradients before loss.backward() calculates new gradients."
        )

        res = engine.query("Why is optimizer.zero_grad() called?", top_k=1)
        self.assertIn("answer", res)
        self.assertGreater(len(res["sources"]), 0)
        self.assertEqual(res["sources"][0]["document_id"], "faq.md")

    def test_file_ingestion(self):
        engine = RAGEngine()
        # Test text file ingestion
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("Batch normalization stabilizes neural network activations across mini-batches.")
            temp_path = f.name

        try:
            chunks = engine.ingest_file(temp_path)
            self.assertGreater(chunks, 0)
            res = engine.query("What does batch normalization do?", top_k=1)
            self.assertIn("answer", res)
            self.assertGreater(len(res["sources"]), 0)
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()

