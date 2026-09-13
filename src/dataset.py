from typing import List, Tuple, Any

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object


class SemanticPairDataset(Dataset):
    """PyTorch Dataset for fine-tuning text embeddings on paired samples."""

    def __init__(self, pairs: List[Tuple[Any, Any, float]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        emb1, emb2, target = self.pairs[idx]
        if HAS_TORCH:
            return (
                torch.tensor(emb1, dtype=torch.float32),
                torch.tensor(emb2, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32)
            )
        return emb1, emb2, target
