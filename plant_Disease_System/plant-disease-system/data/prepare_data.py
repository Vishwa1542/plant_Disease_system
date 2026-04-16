"""
prepare_data.py — Download and split PlantVillage dataset from Kaggle
=======================================================================
Requirements:
  - kaggle API configured (~/.kaggle/kaggle.json)
  - pip install kaggle

Usage:
  python prepare_data.py
"""

import os
import shutil
import random
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
KAGGLE_DATASET = "saroz014/plant-disease"   # Kaggle dataset slug
RAW_DIR = "data/raw"                        # Download destination
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
VAL_SPLIT = 0.2                             # 20% validation
SEED = 42


def download_dataset():
    """Download dataset from Kaggle."""
    print("Downloading PlantVillage dataset from Kaggle...")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {RAW_DIR} --unzip")
    print(f"Download complete → {RAW_DIR}")


def split_dataset(raw_dir: str):
    """
    Split dataset into train/val directories.
    Expected raw structure:
      raw_dir/
        ClassName1/
          img1.jpg
          img2.jpg
        ClassName2/
          ...
    """
    raw_path = Path(raw_dir)

    # Find the actual directory with class folders (may be nested)
    # Look for first directory that contains subdirectories with images
    source_dir = None
    for p in raw_path.rglob("*"):
        if p.is_dir() and any(f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                               for f in p.iterdir() if f.is_file()):
            source_dir = p.parent
            break

    if source_dir is None:
        print("ERROR: Could not find image directories in downloaded dataset.")
        return

    print(f"Source directory found: {source_dir}")
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    print(f"Found {len(class_dirs)} classes.")

    random.seed(SEED)

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + \
                 list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))

        if not images:
            print(f"  Skipping {class_name} — no images found.")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * (1 - VAL_SPLIT))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Copy to train/val
        train_class_dir = Path(TRAIN_DIR) / class_name
        val_class_dir = Path(VAL_DIR) / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        for img in val_images:
            shutil.copy2(img, val_class_dir / img.name)

        print(f"  {class_name}: {len(train_images)} train, {len(val_images)} val")

    print(f"\nData split complete!")
    print(f"  Train → {TRAIN_DIR}")
    print(f"  Val   → {VAL_DIR}")


if __name__ == "__main__":
    download_dataset()
    split_dataset(RAW_DIR)
    print("\nReady to train! Run: python model/train.py")
