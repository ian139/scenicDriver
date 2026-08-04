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
from pathlib import Path
from typing import Optional, Tuple
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

from .inference import get_training_transform, get_inference_transform


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



def create_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    use_resisc45_stats: bool = False
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

    Args:
        data_dir: Path to RESISC45 dataset root directory
        batch_size: Batch size for training (default: 32)
        num_workers: Number of dataloader workers (default: 4)
        val_split: Fraction of data for validation (default: 0.2)
        seed: Random seed for reproducible splits
        use_resisc45_stats: Use RESISC45 specific normalization stats

    Returns:
        Tuple of (train_loader, val_loader)

    Raises:
        ValueError: If data_dir doesn't exist or is empty
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Dataset directory not found: {data_dir}")

    # Check for expected structure
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {data_dir}")

    print(f"Found {len(class_dirs)} class directories")

    # Create transforms
    train_transform = get_training_transform(use_resisc45_stats=use_resisc45_stats)
    # Defensive: strip any ColorJitter to avoid PIL hue overflow issues on some stacks.
    try:
        from torchvision import transforms as _tv

        if isinstance(train_transform, _tv.Compose):
            train_transform.transforms = [
                t for t in train_transform.transforms
                if not isinstance(t, _tv.ColorJitter)
            ]
    except Exception:
        pass
    val_transform = get_inference_transform(use_resisc45_stats=use_resisc45_stats)

    # Load full dataset with train transform (we'll override for val)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

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
        split_point = int(len(indices) * (1 - val_split))
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")

    # Create datasets with appropriate transforms
    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches for stable batch norm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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
    device: str,
    gradient_clip: float = 1.0
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

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for training stability
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
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

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

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
    batch_size: int = 32,
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

    Args:
        data_dir: Path to RESISC45 dataset
        output_dir: Directory to save checkpoints and best model
        epochs: Total training epochs (default: 50)
        batch_size: Training batch size (default: 32)
        learning_rate: Peak learning rate (default: 1e-4)
        weight_decay: AdamW weight decay (default: 0.01)
        warmup_epochs: Number of warmup epochs (default: 5)
        device: Training device (auto-detect if None)
        resume: Path to checkpoint to resume from
        seed: Random seed for reproducibility
        freeze_backbone_epochs: Epochs to freeze backbone (default: 0)
        num_workers: DataLoader workers (default: 4)
        use_resisc45_stats: Use RESISC45 normalization stats
        label_smoothing: Label smoothing factor (default: 0.1)
        gradient_clip: Max gradient norm (default: 1.0)
    """
    # Set seeds for reproducibility
    seed_everything(seed)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

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
    train_loader, val_loader = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        use_resisc45_stats=use_resisc45_stats
    )

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0

    if resume and Path(resume).exists():
        print(f"\nResuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = _best_acc_from_checkpoint(checkpoint)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2%}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training history for logging
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

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
            gradient_clip=gradient_clip
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
            "val_acc": val_acc,
            "best_acc": best_acc,
            "history": history,
        }, output_dir / "checkpoint.pt")

        # Early stopping check - if we hit target accuracy
        if val_acc >= 0.94:
            print(f"\n*** Target accuracy (94%) reached at epoch {epoch + 1}! ***")

    # Save training history
    import json
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
        "--batch_size", type=int, default=32,
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
