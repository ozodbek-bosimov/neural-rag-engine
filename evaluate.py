import time
from typing import List, Dict, Any
from src.rag_engine import RAGEngine


def run_retrieval_benchmark():
    engine = RAGEngine()
    engine.ingest_directory("data/sample_docs")

    test_cases = [
        {
            "query": "How does PyTorch autograd and reverse-mode automatic differentiation work?",
            "expected_doc": "pytorch_deep_learning_guide.md"
        },
        {
            "query": "What are the five essential steps of a standard PyTorch training loop?",
            "expected_doc": "pytorch_deep_learning_guide.md"
        },
        {
            "query": "What is RAG and how does it prevent LLM hallucination?",
            "expected_doc": "retrieval_augmented_generation.md"
        },
        {
            "query": "What mathematical formula defines vector cosine similarity in dense retrieval?",
            "expected_doc": "retrieval_augmented_generation.md"
        },
        {
            "query": "How do Conv2D kernels and max pooling layers extract spatial representations?",
            "expected_doc": "convolutional_neural_networks.md"
        },
        {
            "query": "What are the benefits of weight quantization and SIMD inference for deployment?",
            "expected_doc": "convolutional_neural_networks.md"
        }
    ]

    print("=" * 68)
    print("RAG RETRIEVAL BENCHMARK (Hit Rate @ K, MRR, Latency)")
    print("=" * 68)

    top_k = 3
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks = []
    latencies_ms = []

    for i, item in enumerate(test_cases, 1):
        q = item["query"]
        expected = item["expected_doc"]

        t0 = time.perf_counter()
        res = engine.query(q, top_k=top_k)
        latency = (time.perf_counter() - t0) * 1000
        latencies_ms.append(latency)

        retrieved_ids = [s["document_id"] for s in res["sources"]]
        is_hit_1 = (len(retrieved_ids) > 0 and retrieved_ids[0] == expected)
        if is_hit_1:
            hits_at_1 += 1

        is_hit_k = expected in retrieved_ids
        if is_hit_k:
            hits_at_k += 1
            rank = retrieved_ids.index(expected) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            rank = 0
            reciprocal_ranks.append(0.0)

        status = "PASS" if is_hit_1 else ("NEAR" if is_hit_k else "FAIL")
        print(f"[{status}] Test {i}: '{q[:40]}...' -> Top-1: {retrieved_ids[0] if retrieved_ids else 'None'} ({latency:.2f} ms)")

    total = len(test_cases)
    hit_rate_1 = (hits_at_1 / total) * 100
    hit_rate_k = (hits_at_k / total) * 100
    mrr = sum(reciprocal_ranks) / total
    avg_latency = sum(latencies_ms) / total

    print("-" * 68)
    print("BENCHMARK METRICS:")
    print(f"  • Hit Rate @ 1: {hit_rate_1:.1f}%")
    print(f"  • Hit Rate @ {top_k}: {hit_rate_k:.1f}%")
    print(f"  • MRR (Mean Reciprocal Rank): {mrr:.4f} / 1.0000")
    print(f"  • Mean Latency: {avg_latency:.2f} ms")
    print("=" * 68)

    return {
        "hit_rate_1": hit_rate_1,
        "hit_rate_k": hit_rate_k,
        "mrr": round(mrr, 4),
        "avg_latency_ms": round(avg_latency, 2)
    }


if __name__ == "__main__":
    run_retrieval_benchmark()
