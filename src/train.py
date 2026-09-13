import math
import time
from typing import List, Tuple, Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from src.dataset import SemanticPairDataset
    from src.model import EmbeddingProjectionHead
    HAS_TORCH = True
except ImportError:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from dataset import SemanticPairDataset
        from model import EmbeddingProjectionHead
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False


def generate_synthetic_pairs(num_samples: int = 40, dim: int = 384) -> List[Tuple[List[float], List[float], float]]:
    """Generate synthetic positive and negative vector pairs for contrastive training."""
    data = []
    half = num_samples // 2

    # Positive pairs with small Gaussian-like perturbation
    for i in range(half):
        base = [math.sin(i * 0.1 + j * 0.05) for j in range(dim)]
        norm = math.sqrt(sum(x * x for x in base)) or 1.0
        base = [x / norm for x in base]

        pos = [base[j] + 0.02 * math.cos(j) for j in range(dim)]
        norm_pos = math.sqrt(sum(x * x for x in pos)) or 1.0
        pos = [x / norm_pos for x in pos]
        data.append((base, pos, 1.0))

    # Negative pairs from orthogonal phase space
    for i in range(half):
        v1 = [math.sin(i * 0.2 + j * 0.1) for j in range(dim)]
        v2 = [math.cos(i * 0.3 + j * 0.2) for j in range(dim)]
        norm1 = math.sqrt(sum(x * x for x in v1)) or 1.0
        norm2 = math.sqrt(sum(x * x for x in v2)) or 1.0
        data.append(([x / norm1 for x in v1], [x / norm2 for x in v2], -1.0))

    return data


def run_training_loop(
    dataset_pairs: Optional[List[Tuple[List[float], List[float], float]]] = None,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    save_path: str = "custom_embedding_head.pt"
) -> Optional[Dict[str, Any]]:
    """Standard PyTorch training loop using CosineEmbeddingLoss and AdamW."""
    if not HAS_TORCH:
        print("[!] PyTorch is not installed. Run: pip install torch")
        return None

    if dataset_pairs is None:
        dataset_pairs = generate_synthetic_pairs()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[*] Training device: {device}")

    dataset = SemanticPairDataset(dataset_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EmbeddingProjectionHead(input_dim=384, hidden_dim=256, output_dim=256).to(device)
    criterion = nn.CosineEmbeddingLoss(margin=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    model.train()
    history = []

    print(f"[*] Starting PyTorch training: {epochs} epochs, batch size {batch_size}")
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        total_loss = 0.0

        for batch_x1, batch_x2, target in dataloader:
            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            emb1 = model(batch_x1)
            emb2 = model(batch_x2)
            loss = criterion(emb1, emb2, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        history.append(avg_loss)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.5f} ({elapsed_ms:.1f} ms)")

    torch.save(model.state_dict(), save_path)
    print(f"[✓] Checkpoint saved: {save_path} (Initial: {history[0]:.4f} -> Final: {history[-1]:.4f})")

    return {
        "final_loss": round(history[-1], 5),
        "history": history,
        "model_path": save_path
    }


if __name__ == "__main__":
    run_training_loop(epochs=5)
