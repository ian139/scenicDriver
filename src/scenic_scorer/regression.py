"""Scenic-score regression from classifier and terrain features."""

from pathlib import Path
from typing import Optional, Sequence, Tuple
import numpy as np

import torch
import torch.nn as nn


class ScenicRegressionModel(nn.Module):
    """Predict a scenic score from ViT, terrain, and class features."""

    def __init__(
        self,
        vit_dim: int = 768,
        terrain_dim: int = 6,
        num_classes: int = 45,
        hidden_dim: int = 256
    ):
        super().__init__()

        input_dim = vit_dim + terrain_dim + num_classes

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1], scale to [0, 10]
        )

    def forward(
        self,
        vit_embedding: torch.Tensor,
        terrain_features: torch.Tensor,
        class_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict scenic score.

        Args:
            vit_embedding: [B, 768] ViT features
            terrain_features: [B, 6] terrain metrics
            class_logits: [B, 45] classification logits

        Returns:
            Scenic scores [B, 1] in range [0, 10]
        """
        x = torch.cat([vit_embedding, terrain_features, class_logits], dim=-1)
        score = self.network(x) * 10  # Scale to 0-10
        return score


def _finite_float32(values: np.ndarray, name: str, rank: int) -> np.ndarray:
    """Return ``values`` as finite float32 with the required rank, or raise ``ValueError``."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must be numeric, got dtype {array.dtype}")
    if array.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got shape {array.shape}")
    if rank == 2 and array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one feature column")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = array.astype(np.float32)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} contains non-finite values after float32 conversion")
    return converted


class ScenicScoreDataset(torch.utils.data.Dataset):
    """
    Dataset for training the scenic regression model.

    Each sample contains:
        - vit_embedding: Pre-extracted ViT features
        - terrain_features: [slope_var, elev_change, water_prox, veg_density, coastal, has_water]
        - class_logits: Pre-extracted classification logits
        - scenic_score: Target score (formula-based or human-rated)
        - sample_weight: Training weight per sample (optional; defaults to 1.0)

    Feature arrays must be numeric rank-2 ``[N, D]`` with ``D >= 1``; ``scenic_scores``
    must be numeric rank-1 ``[N]``. All must share ``N`` and be finite after float32
    conversion, otherwise ``ValueError`` is raised.
    """

    def __init__(self, data_path: Path):
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        data = np.load(data_path, allow_pickle=False)
        required = ["vit_embeddings", "terrain_features", "class_logits", "scenic_scores"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Dataset missing required arrays: {missing}")

        features = {
            key: _finite_float32(data[key], key, rank=2)
            for key in ("vit_embeddings", "terrain_features", "class_logits")
        }
        scores = _finite_float32(data["scenic_scores"], "scenic_scores", rank=1)
        n = len(scores)
        if any(len(values) != n for values in features.values()):
            raise ValueError("Dataset arrays have inconsistent lengths")

        self.vit_embeddings = torch.from_numpy(features["vit_embeddings"])
        self.terrain_features = torch.from_numpy(features["terrain_features"])
        self.class_logits = torch.from_numpy(features["class_logits"])
        self.scenic_scores = torch.from_numpy(scores).unsqueeze(-1)
        raw_weights = data["sample_weights"] if "sample_weights" in data else None

        if raw_weights is None:
            weights = np.ones((n,), dtype=np.float32)
        else:
            weights = np.asarray(raw_weights, dtype=np.float32).reshape(-1)
            if len(weights) != n:
                raise ValueError("sample_weights length does not match scenic_scores length")
            if not np.all(np.isfinite(weights)):
                raise ValueError("sample_weights contains non-finite values")
            if np.any(weights <= 0):
                raise ValueError("sample_weights must be strictly positive")
        self.sample_weights = torch.from_numpy(weights).float().unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.scenic_scores)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.vit_embeddings[idx],
            self.terrain_features[idx],
            self.class_logits[idx],
            self.scenic_scores[idx],
            self.sample_weights[idx],
        )


def resolve_device(device: str | None = None) -> str:
    """Resolve the compute device shared by classifier and scenic-score workflows.

    Selection priority when device is None or 'auto': CUDA -> MPS -> CPU.
    Explicit device strings are passed through verbatim.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def validation_split_indices(
    n: int, val_split: float, seed: int
) -> Tuple[list[int], list[int]]:
    """Return the deterministic ``(train, validation)`` membership for ``n`` rows.

    This is the single train/validation split derivation for regression training
    and evaluation. Both sides must consume the same membership so reported
    validation metrics cover exactly the rows the model validated on.
    """
    if n < 1:
        raise ValueError("validation split requires at least one row")
    indices = np.arange(n)
    np.random.RandomState(seed).shuffle(indices)
    split = max(1, int(n * (1 - val_split)))
    train_idx = indices[:split].tolist()
    val_idx = indices[split:].tolist()
    if len(val_idx) == 0:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    return train_idx, val_idx


def train_regression_model(
    data_path: Path,
    output_path: Path,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    val_split: float = 0.15,
    seed: int = 42,
    device: Optional[str] = None,
    weight_decay: float = 1e-4,
    use_sample_weights: bool = True,
    num_workers: int = 4,
    train_indices: Optional[Sequence[int]] = None,
    val_indices: Optional[Sequence[int]] = None,
) -> float:
    """Train the scenic-score regression model and return validation correlation.

    ``train_indices``/``val_indices`` make the validation membership explicit and
    must be provided together: a partition of the dataset rows. When omitted, the
    membership is derived by :func:`validation_split_indices` from
    ``val_split``/``seed``.
    """
    device = resolve_device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = ScenicScoreDataset(Path(data_path))
    n = len(dataset)
    if n < 16:
        raise ValueError("Dataset too small for regression training.")

    if (train_indices is None) != (val_indices is None):
        raise ValueError("train_indices and val_indices must be provided together")
    if train_indices is None:
        train_idx, val_idx = validation_split_indices(n, val_split, seed)
    else:
        train_idx = [int(i) for i in train_indices]
        val_idx = [int(i) for i in val_indices]
        if not train_idx or not val_idx:
            raise ValueError("explicit split requires non-empty train and val indices")
        if (
            set(train_idx) & set(val_idx)
            or set(train_idx) | set(val_idx) != set(range(n))
        ):
            raise ValueError("explicit train/val indices must partition the dataset rows")

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    cuda = str(device).startswith("cuda")
    loader_options = dict(
        batch_size=batch_size, num_workers=num_workers, pin_memory=cuda
    )
    if num_workers:
        loader_options.update(prefetch_factor=2, persistent_workers=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, shuffle=True, **loader_options
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, shuffle=False, **loader_options
    )

    sample_v, sample_t, sample_c, _, _ = dataset[0]
    model = ScenicRegressionModel(
        vit_dim=int(sample_v.shape[0]),
        terrain_dim=int(sample_t.shape[0]),
        num_classes=int(sample_c.shape[0]),
    ).to(device)

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    best_val = float("inf")
    best_corr = -1.0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_corr": [],
    }

    for _ in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            vit_emb, terrain, logits, score, sample_weight = (
                value.to(device, non_blocking=cuda) for value in batch
            )
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=cuda):
                pred = model(vit_emb, terrain, logits)
                per_sample_loss = criterion(pred, score)
                if use_sample_weights:
                    denom = torch.clamp(sample_weight.sum(), min=1e-8)
                    loss = (per_sample_loss * sample_weight).sum() / denom
                else:
                    loss = per_sample_loss.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        model.eval()
        val_loss_sum = 0.0
        val_weight_sum = 0.0
        preds = []
        targets = []
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=cuda):
            for batch in val_loader:
                vit_emb, terrain, logits, score, sample_weight = (
                    value.to(device, non_blocking=cuda) for value in batch
                )
                pred = model(vit_emb, terrain, logits)
                per_sample_loss = criterion(pred.float(), score)
                val_weight = sample_weight if use_sample_weights else torch.ones_like(score)
                val_loss_sum += float((per_sample_loss * val_weight).sum().item())
                val_weight_sum += float(val_weight.sum().item())
                preds.append(pred.float().cpu().numpy().reshape(-1))
                targets.append(score.float().cpu().numpy().reshape(-1))

        # Aggregate over samples (weighted when enabled) so uneven final batches are not over-weighted.
        val_loss = val_loss_sum / val_weight_sum if val_weight_sum > 0 else 0.0
        all_preds = np.concatenate(preds) if preds else np.array([0.0], dtype=np.float32)
        all_targets = np.concatenate(targets) if targets else np.array([0.0], dtype=np.float32)
        corr = float(np.corrcoef(all_preds, all_targets)[0, 1]) if len(all_preds) > 1 else 0.0
        if not np.isfinite(corr):
            corr = 0.0

        history["train_loss"].append(float(np.mean(train_losses)) if train_losses else 0.0)
        history["val_loss"].append(val_loss)
        history["val_corr"].append(corr)

        if val_loss < best_val:
            best_val = val_loss
            best_corr = corr
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val,
                    "best_val_corr": best_corr,
                    "vit_dim": int(sample_v.shape[0]),
                    "terrain_dim": int(sample_t.shape[0]),
                    "num_classes": int(sample_c.shape[0]),
                    "used_sample_weights": bool(use_sample_weights),
                    "history": history,
                },
                output_path,
            )

    return best_corr


