#!/usr/bin/env python3
"""
Landscape Classification Demo Script
Owner: progno-ml-vision agent

Demonstrates the landscape classifier on sample images.
Can use either a trained model or pretrained ViT for zero-shot testing.

Usage:
    # Classify images in data/tiles/ using pretrained ViT
    python scripts/classify_demo.py

    # Use a trained model
    python scripts/classify_demo.py --model models/classifier/best_model.pt

    # Classify specific images
    python scripts/classify_demo.py --images image1.jpg image2.png

    # Classify all images in a directory
    python scripts/classify_demo.py --directory path/to/images

    # Show top-5 predictions for each image
    python scripts/classify_demo.py --top 5

    # Save results to JSON
    python scripts/classify_demo.py --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_sample_images(
    tiles_dir: Optional[Path] = None,
    limit: int = 10
) -> List[Path]:
    """Find sample images to classify."""
    if tiles_dir is None:
        tiles_dir = PROJECT_ROOT / "data" / "tiles"

    if not tiles_dir.exists():
        print(f"Tiles directory not found: {tiles_dir}")
        return []

    # Find all image files
    extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    images = []
    for ext in extensions:
        images.extend(tiles_dir.glob(f"*{ext}"))
        images.extend(tiles_dir.glob(f"*{ext.upper()}"))

    # Also check subdirectories
    for ext in extensions:
        images.extend(tiles_dir.glob(f"**/*{ext}"))

    # Remove duplicates and sort
    images = sorted(set(images))

    if limit:
        images = images[:limit]

    return images


def format_scenic_bar(weight: float, width: int = 20) -> str:
    """Create a visual bar for scenic weight."""
    filled = int(weight * width)
    empty = width - filled

    if weight >= 0.7:
        color = "\033[92m"  # Green
    elif weight >= 0.4:
        color = "\033[93m"  # Yellow
    else:
        color = "\033[91m"  # Red

    reset = "\033[0m"
    return f"{color}{'█' * filled}{'░' * empty}{reset} {weight:.0%}"


def classify_and_display(
    images: List[Path],
    model_path: Optional[Path] = None,
    top_k: int = 1,
    output_file: Optional[Path] = None,
    show_scenic: bool = True,
    verbose: bool = False
):
    """
    Classify images and display results.

    Args:
        images: List of image paths to classify
        model_path: Path to trained model weights (None for pretrained)
        top_k: Number of top predictions to show
        output_file: Optional path to save JSON results
        show_scenic: Whether to show scenic weight bars
        verbose: Show additional details
    """
    # Import classifier modules
    try:
        from src.classifier.inference import (
            load_model,
            classify_image,
            batch_classify,
            compute_scenic_score,
            get_class_distribution
        )
        from src.classifier.model import TERRAIN_CLASSES, get_scenic_weight
    except ImportError as e:
        print(f"Error importing classifier modules: {e}")
        print("\nMake sure you're in the project root and dependencies are installed:")
        print("  pip install torch torchvision timm")
        return 1

    print("\n" + "=" * 70)
    print("Landscape Classification Demo")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    if model_path and Path(model_path).exists():
        print(f"  Using trained model: {model_path}")
        model = load_model(model_path=model_path)
    else:
        print("  Using pretrained ViT (no fine-tuning)")
        print("  Note: Classification may not match RESISC45 labels without training")
        model = load_model(pretrained=True)

    model_info = model.get_model_info()
    print(f"  Architecture: {model_info['architecture']}")
    print(f"  Parameters: {model_info['total_params']:,}")
    print(f"  Device: {model_info['device']}")

    print(f"\nClassifying {len(images)} images...")
    print("-" * 70)

    # Classify all images
    all_results = []

    for i, image_path in enumerate(images, 1):
        try:
            result = classify_image(image_path, model)
            result["image_path"] = str(image_path)
            result["filename"] = image_path.name
            all_results.append(result)

            # Display result
            print(f"\n[{i}/{len(images)}] {image_path.name}")

            if top_k == 1:
                print(f"  Class: {result['class']}")
                print(f"  Confidence: {result['confidence']:.1%}")
                if show_scenic:
                    print(f"  Scenic: {format_scenic_bar(result['scenic_weight'])}")
            else:
                print(f"  Top {top_k} predictions:")
                for rank, (cls, prob) in enumerate(result['top_5'][:top_k], 1):
                    scenic = get_scenic_weight(cls)
                    if show_scenic:
                        print(f"    {rank}. {cls}: {prob:.1%} (scenic: {scenic:.0%})")
                    else:
                        print(f"    {rank}. {cls}: {prob:.1%}")

            if verbose:
                print(f"  Class index: {result['class_idx']}")

        except Exception as e:
            print(f"\n[{i}/{len(images)}] {image_path.name}")
            print(f"  ERROR: {e}")
            all_results.append({
                "image_path": str(image_path),
                "filename": image_path.name,
                "error": str(e)
            })

    # Summary statistics
    valid_results = [r for r in all_results if "error" not in r]

    if valid_results:
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)

        # Class distribution
        print("\nClass distribution:")
        distribution = get_class_distribution(valid_results, top_n=5)
        for cls, count, pct in distribution:
            scenic = get_scenic_weight(cls)
            print(f"  {cls}: {count} images ({pct:.0%}) - scenic: {scenic:.0%}")

        # Scenic scores
        if show_scenic:
            print("\nScenic scores:")
            scenic_mean = compute_scenic_score(valid_results, "mean")
            scenic_weighted = compute_scenic_score(valid_results, "weighted_mean")
            scenic_max = compute_scenic_score(valid_results, "max")

            print(f"  Average scenic weight: {scenic_mean:.1%}")
            print(f"  Confidence-weighted:   {scenic_weighted:.1%}")
            print(f"  Maximum scenic:        {scenic_max:.1%}")

        # Confidence stats
        confidences = [r["confidence"] for r in valid_results]
        print(f"\nConfidence statistics:")
        print(f"  Mean: {sum(confidences)/len(confidences):.1%}")
        print(f"  Min:  {min(confidences):.1%}")
        print(f"  Max:  {max(confidences):.1%}")

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for r in all_results:
            jr = {k: v for k, v in r.items() if k != "all_probs"}
            if "all_probs" in r:
                jr["all_probs"] = r["all_probs"].tolist()
            json_results.append(jr)

        with open(output_path, "w") as f:
            json.dump({
                "model_path": str(model_path) if model_path else "pretrained",
                "num_images": len(images),
                "results": json_results,
            }, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    print("\nDone!")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate landscape classification on sample images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Classify sample tiles with pretrained model
    python scripts/classify_demo.py

    # Use a trained model
    python scripts/classify_demo.py --model models/classifier/best_model.pt

    # Classify specific images
    python scripts/classify_demo.py --images path/to/image1.jpg path/to/image2.png

    # Classify all images in a directory
    python scripts/classify_demo.py --directory path/to/images

    # Show top-5 predictions
    python scripts/classify_demo.py --top 5

    # Save results to JSON
    python scripts/classify_demo.py --output results.json

    # Limit number of images
    python scripts/classify_demo.py --limit 5
"""
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--images", "-i",
        nargs="+",
        type=Path,
        help="Specific image files to classify"
    )
    input_group.add_argument(
        "--directory", "-d",
        type=Path,
        help="Directory containing images to classify"
    )

    # Model options
    parser.add_argument(
        "--model", "-m",
        type=Path,
        default=None,
        help="Path to trained model weights (uses pretrained ViT if not provided)"
    )

    # Output options
    parser.add_argument(
        "--top", "-k",
        type=int,
        default=1,
        help="Show top-k predictions (default: 1)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Maximum number of images to process (default: 10)"
    )
    parser.add_argument(
        "--no-scenic",
        action="store_true",
        help="Don't show scenic weight information"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )

    args = parser.parse_args()

    # Determine images to classify
    if args.images:
        images = [Path(p) for p in args.images if Path(p).exists()]
        if len(images) < len(args.images):
            missing = set(args.images) - set(images)
            print(f"Warning: Some images not found: {missing}")
    elif args.directory:
        if not args.directory.exists():
            print(f"Directory not found: {args.directory}")
            return 1
        images = find_sample_images(args.directory, limit=args.limit)
    else:
        # Default: look for tiles in data/tiles/
        images = find_sample_images(limit=args.limit)

    if not images:
        print("No images found to classify.")
        print("\nTo download sample tiles, run:")
        print("  python scripts/download_sample_tiles.py")
        print("\nOr specify images manually:")
        print("  python scripts/classify_demo.py --images path/to/image.jpg")
        return 1

    # Run classification
    return classify_and_display(
        images=images,
        model_path=args.model,
        top_k=args.top,
        output_file=args.output,
        show_scenic=not args.no_scenic,
        verbose=args.verbose
    )


if __name__ == "__main__":
    sys.exit(main())
