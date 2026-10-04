"""
Training script for landscape classifier
Owner: progno-ml-vision agent

Usage:
    python -m src.classifier.train --data_dir ./data/resisc45 --epochs 50

    # With custom settings
    python -m src.classifier.train --data_dir ./data/resisc45 --epochs 100 --batch_size 16 --lr 5e-5

    # Resume from checkpoint
    python -m src.classifier.train --data_dir ./data/resisc45 --resume models/classifier/checkpoint.pt

Target: 94%+ accuracy on RESISC45 test set
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Tuple, Union
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

from src.scenic_scorer.regression import resolve_device

from .inference import get_training_transform, get_inference_transform
from .model import TERRAIN_CLASSES


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _best_acc_from_checkpoint(checkpoint: dict) -> float:
    """Read the canonical best metric required by current checkpoints."""
    if "best_acc" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing required 'best_acc'; "
            "resume checkpoints must be written by the current classifier trainer"
        )
    return float(checkpoint["best_acc"])


_HISTORY_KEYS = ("train_loss", "val_loss", "val_acc", "lr")


def _history_from_checkpoint(checkpoint: dict) -> dict:
    """
    Read the canonical metric history required by current checkpoints.

    Raises:
        ValueError: If 'history' is missing, not a dict, or lacks a required list
    """
    if "history" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing required 'history'; "
            "resume checkpoints must be written by the current classifier trainer"
        )
    history = checkpoint["history"]
    if not isinstance(history, dict):
        raise ValueError(
            "Checkpoint 'history' must be a dict of metric lists; "
            "resume checkpoints must be written by the current classifier trainer"
        )
    for key in _HISTORY_KEYS:
        if not isinstance(history.get(key), list):
            raise ValueError(
                f"Checkpoint 'history' is missing required list '{key}'; "
                "resume checkpoints must be written by the current classifier trainer"
            )
    return history


def _verify_split_identity(checkpoint: dict, split_identity: dict) -> None:
    """
    Fail resume unless the checkpoint records the run's dataset split identity.

    Raises:
        ValueError: If 'split_identity' is missing or differs from the split
            created for this run
    """
    if "split_identity" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing required 'split_identity'; "
            "resume checkpoints must be written by the current classifier trainer"
        )
    recorded = checkpoint["split_identity"]
    if recorded != split_identity:
        raise ValueError(
            "Checkpoint 'split_identity' does not match the dataset split created "
            f"for this run; checkpoint={recorded!r}, current={split_identity!r}. "
            "Resume against the same dataset contents and seed, or train from scratch"
        )



def create_dataloaders(
    data_dir: Path,
    batch_size: int = 256,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    use_resisc45_stats: bool = False,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for RESISC45.

    RESISC45 structure:
        data_dir/
            airplane/
                airplane_001.jpg
                airplane_002.jpg
                ...
            airport/
                airport_001.jpg
                ...
            ... (45 classes total, 700 images each)

    The dataset is split into train/val maintaining class balance.
    Training uses augmentation, validation uses only resize + normalize.

    Each class keeps at least one training image; singleton classes are train-only.
    Classes with two or more images also keep at least one validation image.
    A dataset must provide at least one validation image overall. For training
    sets smaller than one batch, the partial batch is retained rather than dropped.

    Args:
        data_dir: Path to RESISC45 dataset root directory
        batch_size: Batch size for training (default: 256)
        num_workers: Number of dataloader workers (default: 4)
        pin_memory: Enable page-locked host buffers for CUDA transfers
        val_split: Fraction of data for validation (default: 0.2)
        seed: Random seed for reproducible splits
        use_resisc45_stats: Use RESISC45 specific normalization stats

    Returns:
        Tuple of (train_loader, val_loader)

    Raises:
        ValueError: If the dataset/classes/split cannot support training and validation
    """
    data_dir = Path(data_dir)
    if not 0 < val_split < 1:
        raise ValueError("val_split must be strictly between 0 and 1")
    if not data_dir.exists():
        raise ValueError(f"Dataset directory not found: {data_dir}")

    # Check for expected structure
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {data_dir}")

    print(f"Found {len(class_dirs)} class directories")

    # Create transforms
    train_transform = get_training_transform(use_resisc45_stats=use_resisc45_stats)
    val_transform = get_inference_transform(use_resisc45_stats=use_resisc45_stats)

    # Load full dataset with train transform (we'll override for val)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    if full_dataset.classes != TERRAIN_CLASSES:
        raise ValueError(
            "Dataset class order must match classifier TERRAIN_CLASSES: "
            f"expected {TERRAIN_CLASSES}, got {full_dataset.classes}"
        )

    print(f"Total images: {len(full_dataset)}")
    print(f"Classes: {full_dataset.classes[:5]}... ({len(full_dataset.classes)} total)")

    # Create stratified train/val split
    # Get indices for each class
    class_indices = {}
    for idx, (_, label) in enumerate(full_dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Split each class proportionally
    train_indices = []
    val_indices = []

    rng = np.random.RandomState(seed)
    for label, indices in class_indices.items():
        rng.shuffle(indices)
        split_point = max(1, int(len(indices) * (1 - val_split)))
        if len(indices) > 1:
            split_point = min(split_point, len(indices) - 1)
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    if not val_indices:
        raise ValueError("Dataset needs at least one class with two images for validation")

    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")

    # Create datasets with appropriate transforms
    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_transform)

    # Create dataloaders
    effective_pin = pin_memory
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": effective_pin,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=len(train_dataset) >= batch_size,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


class TransformSubset(Dataset):
    """
    A subset of a dataset with a custom transform.

    This allows us to use different transforms for train and val
    while sharing the underlying ImageFolder dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: list,
        transform: transforms.Compose
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        # Get original sample
        original_idx = self.indices[idx]
        path, label = self.dataset.samples[original_idx]

        # Load and transform image
        from PIL import Image
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: Union[str, torch.device],
    gradient_clip: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        gradient_clip: Maximum gradient norm for clipping

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    device_obj = torch.device(device) if isinstance(device, str) else device
    is_cuda = (device_obj.type == "cuda")

    active_scaler: torch.amp.GradScaler = (
        scaler
        if scaler is not None
        else torch.amp.GradScaler("cuda", enabled=is_cuda)
    )

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images, labels = batch
        images = images.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=is_cuda):
            outputs = model(images)
            loss = criterion(outputs, labels)

        active_scaler.scale(loss).backward()

        # Gradient clipping for training stability
        if gradient_clip > 0:
            active_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        active_scaler.step(optimizer)
        active_scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: Union[str, torch.device]
) -> Tuple[float, float]:
    """
    Validate model on the validation set.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to run validation on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    device_obj = torch.device(device) if isinstance(device, str) else device
    is_cuda = (device_obj.type == "cuda")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = batch
            images = images.to(device, non_blocking=is_cuda)
            labels = labels.to(device, non_blocking=is_cuda)

            with torch.amp.autocast("cuda", enabled=is_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            num_batches += 1
    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, accuracy


def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_epochs: int = 5,
    device: Optional[str] = None,
    resume: Optional[Path] = None,
    seed: int = 42,
    freeze_backbone_epochs: int = 0,
    num_workers: int = 4,
    use_resisc45_stats: bool = False,
    label_smoothing: float = 0.1,
    gradient_clip: float = 1.0,
    pin_memory: Optional[bool] = None,
) -> None:
    """
    Train the landscape classifier on RESISC45 dataset.

    Target: 94%+ accuracy on RESISC45 validation set

    Training strategy:
    1. Use pretrained ViT-B/16 from ImageNet
    2. Optionally freeze backbone for initial epochs
    3. Use AdamW with cosine annealing + warmup
    4. Apply label smoothing for better generalization
    5. Use gradient clipping for stability

    Checkpoints preserve optimizer, scheduler, scaler, and metric history across
    resume. Split identity records the seed and a SHA256 manifest of class order,
    relative image paths, labels, and train/validation membership (not file contents).
    Resume verifies the checkpoint's split identity against the split created from
    the dataset on disk and fails when they differ.

    Args:
        data_dir: Path to RESISC45 dataset
        output_dir: Directory to save checkpoints and best model
        epochs: Total training epochs (default: 50)
        batch_size: Training batch size (default: 256)
        learning_rate: Peak learning rate (default: 1e-4)
        weight_decay: AdamW weight decay (default: 0.01)
        warmup_epochs: Number of warmup epochs (default: 5)
        device: Training device (auto-detect if None)
        resume: Path to checkpoint to resume from
        seed: Random seed for reproducibility
        num_workers: DataLoader workers (default: 4)
        pin_memory: Enable pinned host buffers for CUDA transfers
        use_resisc45_stats: Use RESISC45 normalization stats
        label_smoothing: Label smoothing factor (default: 0.1)
        gradient_clip: Max gradient norm (default: 1.0)
    """
    if resume is not None and not Path(resume).is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")

    # Set seeds for reproducibility
    seed_everything(seed)

    if device is None:
        device = resolve_device()

    print("=" * 60)
    print("Landscape Classifier Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Freeze backbone epochs: {freeze_backbone_epochs}")
    print(f"Label smoothing: {label_smoothing}")
    print("=" * 60)

    # Create model
    from .model import LandscapeClassifier
    model = LandscapeClassifier(
        num_classes=45,
        pretrained=True,
        device=device
    )
    model = model.to(device)

    model_info = model.get_model_info()
    print(f"\nModel: {model_info['architecture']}")
    print(f"Parameters: {model_info['total_params']:,}")
    print(f"Trainable: {model_info['trainable_params']:,}")

    # Create dataloaders
    print(f"\nLoading dataset from {data_dir}...")
    device_obj = torch.device(device)
    is_cuda = (device_obj.type == "cuda")
    if pin_memory is None:
        pin_memory = is_cuda
    pin_memory = bool(pin_memory)

    train_loader, val_loader = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        use_resisc45_stats=use_resisc45_stats,
        pin_memory=pin_memory,
    )
    dataset = train_loader.dataset.dataset
    split_manifest = {
        "classes": dataset.classes,
        "train": [
            (Path(dataset.samples[i][0]).relative_to(data_dir).as_posix(), dataset.samples[i][1])
            for i in train_loader.dataset.indices
        ],
        "validation": [
            (Path(dataset.samples[i][0]).relative_to(data_dir).as_posix(), dataset.samples[i][1])
            for i in val_loader.dataset.indices
        ],
    }
    split_identity = {
        "seed": seed,
        "sha256": hashlib.sha256(
            json.dumps(split_manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    # Use different learning rates for backbone vs classifier
    backbone_params = list(model.backbone.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": learning_rate * 0.1},  # Lower LR for pretrained
        {"params": classifier_params, "lr": learning_rate},
    ], weight_decay=weight_decay)

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    if resume is not None:
        print(f"\nResuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        _verify_split_identity(checkpoint, split_identity)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = _best_acc_from_checkpoint(checkpoint)
        history = _history_from_checkpoint(checkpoint)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2%}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Freeze backbone if requested
    if freeze_backbone_epochs > 0 and start_epoch < freeze_backbone_epochs:
        print(f"\nFreezing backbone for first {freeze_backbone_epochs} epochs")
        for param in model.backbone.parameters():
            param.requires_grad = False

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(start_epoch, epochs):
        # Unfreeze backbone after freeze period
        if epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
            print("\nUnfreezing backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = True

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{epochs} (lr: {current_lr:.2e})")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip=gradient_clip, scaler=scaler
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Log metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_weights(output_dir / "best_model.pt")
            print(f"*** New best accuracy: {best_acc:.2%} ***")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_acc": val_acc,
            "best_acc": best_acc,
            "history": history,
            "split_identity": split_identity,
        }, output_dir / "checkpoint.pt")

        # Early stopping check - if we hit target accuracy
        if val_acc >= 0.94:
            print(f"\n*** Target accuracy (94%) reached at epoch {epoch + 1}! ***")

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation accuracy: {best_acc:.2%}")
    print(f"Best model saved to: {output_dir / 'best_model.pt'}")
    print(f"Training history saved to: {output_dir / 'training_history.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train landscape classifier on RESISC45 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--data_dir", type=Path, required=True,
        help="Path to RESISC45 dataset directory"
    )

    # Optional arguments
    parser.add_argument(
        "--output_dir", type=Path, default=Path("./models/classifier"),
        help="Directory to save checkpoints and best model"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Total training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Peak learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="AdamW weight decay"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5,
        help="Number of warmup epochs"
    )
    parser.add_argument(
        "--freeze_backbone", type=int, default=0,
        help="Epochs to freeze backbone (0 = no freezing)"
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--use_resisc45_stats", action="store_true",
        help="Use RESISC45 normalization stats instead of ImageNet"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1,
        help="Label smoothing factor"
    )
    parser.add_argument(
        "--gradient_clip", type=float, default=1.0,
        help="Max gradient norm for clipping"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
        help="Training device"
    )

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        device=None if args.device == "auto" else args.device,
        resume=args.resume,
        seed=args.seed,
        freeze_backbone_epochs=args.freeze_backbone,
        num_workers=args.num_workers,
        use_resisc45_stats=args.use_resisc45_stats,
        label_smoothing=args.label_smoothing,
        gradient_clip=args.gradient_clip,
    )
