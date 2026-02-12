#!/usr/bin/env python3
"""
Download RESISC45 Dataset Script
Owner: progno-ml-vision agent

RESISC45 (Remote Sensing Image Scene Classification) dataset contains
45 scene classes with 700 images per class (31,500 total images).
Image size: 256x256 RGB

Dataset paper: https://arxiv.org/abs/1703.00121
TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/resisc45

Usage:
    python scripts/download_resisc45.py

    # Specify output directory
    python scripts/download_resisc45.py --output data/resisc45

    # Extract only (if already downloaded)
    python scripts/download_resisc45.py --extract-only data/NWPU-RESISC45.rar

Notes:
    The official dataset is distributed as a RAR archive.
    This script provides instructions for downloading from various sources.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Dataset information
RESISC45_INFO = {
    "name": "NWPU-RESISC45",
    "num_classes": 45,
    "images_per_class": 700,
    "total_images": 31500,
    "image_size": (256, 256, 3),
    "format": "JPEG",
    "archive_name": "NWPU-RESISC45.rar",
    "extracted_name": "NWPU-RESISC45",
}

# Known download sources
DOWNLOAD_SOURCES = [
    {
        "name": "OneDrive (Official)",
        "url": "https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs",
        "notes": "Official source from the dataset authors",
        "auto": False,  # Requires manual download
    },
    {
        "name": "Google Drive (Mirror)",
        "url": "https://drive.google.com/file/d/1DnPSU5nVSN7xv95bpZ3XQ0JhKXZOKgIv",
        "notes": "Community mirror - may be removed",
        "auto": False,
    },
    {
        "name": "Kaggle",
        "url": "https://www.kaggle.com/datasets/apollo2506/nwpu-resisc45-dataset",
        "notes": "Requires Kaggle account and API credentials",
        "auto": True,
        "command": "kaggle datasets download -d apollo2506/nwpu-resisc45-dataset",
    },
    {
        "name": "TensorFlow Datasets",
        "url": "https://www.tensorflow.org/datasets/catalog/resisc45",
        "notes": "Auto-download via tfds.load('resisc45')",
        "auto": True,
    },
]

# Classes in RESISC45 (alphabetically sorted)
RESISC45_CLASSES = [
    "airplane", "airport", "baseball_diamond", "basketball_court", "beach",
    "bridge", "chaparral", "church", "circular_farmland", "cloud",
    "commercial_area", "dense_residential", "desert", "forest", "freeway",
    "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection",
    "island", "lake", "meadow", "medium_residential", "mobile_home_park",
    "mountain", "overpass", "palace", "parking_lot", "railway",
    "railway_station", "rectangular_farmland", "river", "roundabout", "runway",
    "sea_ice", "ship", "snowberg", "sparse_residential", "stadium",
    "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland"
]


def check_unrar():
    """Check if unrar is available."""
    try:
        subprocess.run(["unrar", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_kaggle():
    """Check if Kaggle CLI is configured."""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        # Check for credentials
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        return kaggle_json.exists()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_rar(rar_path: Path, output_dir: Path) -> bool:
    """Extract RAR archive."""
    if not check_unrar():
        print("\nERROR: unrar not found. Install it:")
        print("  macOS:   brew install unrar")
        print("  Ubuntu:  sudo apt install unrar")
        print("  Windows: Download from https://www.rarlab.com/download.htm")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting {rar_path}...")
    try:
        subprocess.run(
            ["unrar", "x", "-o+", str(rar_path), str(output_dir)],
            check=True
        )
        print("Extraction complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Extraction failed: {e}")
        return False


def download_from_kaggle(output_dir: Path) -> bool:
    """Download dataset from Kaggle."""
    if not check_kaggle():
        print("\nKaggle CLI not configured. Setup instructions:")
        print("1. Install: pip install kaggle")
        print("2. Go to https://www.kaggle.com/account")
        print("3. Click 'Create New Token' to download kaggle.json")
        print("4. Move to ~/.kaggle/kaggle.json")
        print("5. chmod 600 ~/.kaggle/kaggle.json")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nDownloading from Kaggle...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "apollo2506/nwpu-resisc45-dataset",
             "-p", str(output_dir)],
            check=True
        )

        # Extract the zip file
        import zipfile
        zip_path = output_dir / "nwpu-resisc45-dataset.zip"
        if zip_path.exists():
            print("Extracting zip file...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            zip_path.unlink()  # Remove zip after extraction

        print("Download complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return False


def download_with_tfds(output_dir: Path) -> bool:
    """Download dataset using TensorFlow Datasets."""
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        print("\nTensorFlow Datasets not installed. Install with:")
        print("  pip install tensorflow-datasets")
        return False

    print("\nDownloading via TensorFlow Datasets...")
    print("Note: This downloads to TensorFlow's default data directory.")

    try:
        # Download the dataset
        ds, info = tfds.load(
            'resisc45',
            split='train',
            with_info=True,
            data_dir=str(output_dir / 'tfds_data')
        )
        print(f"Downloaded {info.splits['train'].num_examples} images")

        # Convert to ImageFolder format for PyTorch compatibility
        print("\nConverting to ImageFolder format...")
        convert_tfds_to_imagefolder(ds, info, output_dir)

        print("Conversion complete!")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def convert_tfds_to_imagefolder(ds, info, output_dir: Path):
    """Convert TensorFlow Dataset to ImageFolder structure."""
    from PIL import Image
    import numpy as np

    class_names = info.features['label'].names
    counts = {name: 0 for name in class_names}

    for example in ds:
        image = example['image'].numpy()
        label = example['label'].numpy()
        class_name = class_names[label]

        # Create class directory
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        counts[class_name] += 1
        filename = f"{class_name}_{counts[class_name]:03d}.jpg"
        img = Image.fromarray(image)
        img.save(class_dir / filename)

    print(f"Saved {sum(counts.values())} images to {output_dir}")


def verify_dataset(data_dir: Path) -> bool:
    """Verify the dataset structure and counts."""
    print(f"\nVerifying dataset at {data_dir}...")

    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return False

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if len(class_dirs) == 0:
        print("No class directories found!")
        return False

    print(f"Found {len(class_dirs)} class directories")

    # Check each class
    issues = []
    total_images = 0

    for class_dir in class_dirs:
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
        images += list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.JPEG"))

        if len(images) == 0:
            issues.append(f"  {class_dir.name}: No images found")
        elif len(images) < 700:
            issues.append(f"  {class_dir.name}: Only {len(images)} images (expected 700)")

        total_images += len(images)

    if issues:
        print("\nWarnings:")
        for issue in issues[:10]:  # Show first 10 issues
            print(issue)
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")

    print(f"\nTotal images: {total_images}")
    print(f"Expected: {RESISC45_INFO['total_images']}")

    if total_images >= RESISC45_INFO['total_images'] * 0.9:  # Allow 10% tolerance
        print("\nDataset verification: PASSED")
        return True
    else:
        print("\nDataset verification: FAILED")
        return False


def print_download_instructions():
    """Print manual download instructions."""
    print("\n" + "=" * 70)
    print("RESISC45 Dataset Download Instructions")
    print("=" * 70)

    print(f"""
Dataset: {RESISC45_INFO['name']}
- {RESISC45_INFO['num_classes']} scene categories
- {RESISC45_INFO['images_per_class']} images per category
- {RESISC45_INFO['total_images']} total images
- Image size: {RESISC45_INFO['image_size'][0]}x{RESISC45_INFO['image_size'][1]} RGB
""")

    print("Download Options:")
    print("-" * 70)

    for i, source in enumerate(DOWNLOAD_SOURCES, 1):
        print(f"\n{i}. {source['name']}")
        print(f"   URL: {source['url']}")
        print(f"   Notes: {source['notes']}")
        if source.get('command'):
            print(f"   Command: {source['command']}")

    print("\n" + "-" * 70)
    print("""
RECOMMENDED: Manual download from OneDrive (Option 1)

Steps:
1. Go to the OneDrive link above
2. Download NWPU-RESISC45.rar (~400 MB)
3. Run this script with --extract-only:

   python scripts/download_resisc45.py --extract-only /path/to/NWPU-RESISC45.rar

The dataset will be extracted to data/resisc45/ in ImageFolder format.
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare RESISC45 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show download instructions
    python scripts/download_resisc45.py

    # Download via Kaggle
    python scripts/download_resisc45.py --source kaggle

    # Download via TensorFlow Datasets
    python scripts/download_resisc45.py --source tfds

    # Extract a manually downloaded RAR file
    python scripts/download_resisc45.py --extract-only ~/Downloads/NWPU-RESISC45.rar

    # Verify existing dataset
    python scripts/download_resisc45.py --verify data/resisc45
"""
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./data/resisc45"),
        help="Output directory for the dataset"
    )
    parser.add_argument(
        "--source",
        choices=["kaggle", "tfds", "manual"],
        default="manual",
        help="Download source"
    )
    parser.add_argument(
        "--extract-only",
        type=Path,
        metavar="RAR_FILE",
        help="Extract an existing RAR file"
    )
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="DATA_DIR",
        help="Verify an existing dataset directory"
    )

    args = parser.parse_args()

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    print("\n" + "=" * 70)
    print("RESISC45 Dataset Downloader")
    print("=" * 70)

    # Verify mode
    if args.verify:
        verify_dataset(args.verify)
        return 0

    # Extract mode
    if args.extract_only:
        if not args.extract_only.exists():
            print(f"RAR file not found: {args.extract_only}")
            return 1

        if extract_rar(args.extract_only, args.output):
            # Find the extracted directory
            extracted = args.output / RESISC45_INFO['extracted_name']
            if extracted.exists():
                print(f"\nDataset extracted to: {extracted}")
                verify_dataset(extracted)
            return 0
        return 1

    # Download mode
    if args.source == "manual":
        print_download_instructions()
        return 0

    elif args.source == "kaggle":
        if download_from_kaggle(args.output):
            verify_dataset(args.output)
            return 0
        return 1

    elif args.source == "tfds":
        if download_with_tfds(args.output):
            verify_dataset(args.output)
            return 0
        return 1


if __name__ == "__main__":
    sys.exit(main())
