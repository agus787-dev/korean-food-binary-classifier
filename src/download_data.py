"""Download Food-101 and build train/val/test folders for the selected classes in dataset/.

The official test set is kept as it is. A validation set is taken from the official train set.
"""
import random
import shutil
from pathlib import Path

from torchvision.datasets import Food101

from utils import DATA_DIR, DATASET_DIR, TRAIN_DIR, TEST_DIR, VAL_DIR

KEEP_CLASSES = ["bibimbap", "fried_rice"]
SPLITS = ("train", "val", "test")
SEED = 42
VAL_RATIO = 0.2  # 20% of the official train images become validation images


def download_food101(split: str) -> Food101:
    """Load Food-101. Skips the download if the dataset already exists in DATA_DIR."""
    return Food101(root=DATA_DIR, split=split, download=True)


def collect_class_images(ds: Food101, keep: list) -> dict:
    """Return {class_name: sorted list of image paths} for the selected classes."""
    missing = [c for c in keep if c not in ds.classes]
    if missing:
        raise ValueError(f"Unknown classes: {missing}")

    images = {cls: [] for cls in keep}
    for img_path, label in zip(ds._image_files, ds._labels):
        cls = ds.classes[label]
        if cls in images:
            images[cls].append(Path(img_path))

    # Sort so the result does not depend on the OS file order
    return {cls: sorted(paths) for cls, paths in images.items()}


def split_train_val(image_paths: list, val_ratio: float = VAL_RATIO):
    """Split one class into train and validation files.

    The list must be shuffled before calling this function.
    """
    val_count = int(len(image_paths) * val_ratio)
    train_count = len(image_paths) - val_count
    return image_paths[:train_count], image_paths[train_count:]


def copy_files(file_paths: list, destination_dir: Path) -> None:
    """Copy image files into the destination class folder."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    for file_path in file_paths:
        shutil.copy(file_path, destination_dir / file_path.name)


def clear_split_folders(out_root: Path) -> None:
    """Remove old train/val/test folders so we can rebuild a clean split.

    Only these three folders are removed. The raw Food-101 data in DATA_DIR is never touched.
    """
    for split in SPLITS:
        split_dir = out_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)


def print_counts(out_root: Path, keep: list) -> None:
    """Print how many images each split/class folder contains."""
    for split in SPLITS:
        for cls in keep:
            n = len(list((out_root / split / cls).glob("*.jpg")))
            print(f"{split:5s} | {cls:11s} | {n} images")


def main() -> None:
    train_ds = download_food101("train")
    test_ds = download_food101("test")
    print(f"[train] full dataset size: {len(train_ds)} images")
    print(f"[test] full dataset size: {len(test_ds)} images")

    train_images = collect_class_images(train_ds, KEEP_CLASSES)
    test_images = collect_class_images(test_ds, KEEP_CLASSES)

    # Local random generator: the same SEED always gives the same split
    rng = random.Random(SEED)

    clear_split_folders(DATASET_DIR)

    for cls in KEEP_CLASSES:
        image_paths = list(train_images[cls])
        rng.shuffle(image_paths)

        train_files, val_files = split_train_val(image_paths)

        copy_files(train_files, TRAIN_DIR / cls)
        copy_files(val_files, VAL_DIR / cls)
        copy_files(test_images[cls], TEST_DIR / cls)

    print("\nFinal counts:")
    print_counts(DATASET_DIR, KEEP_CLASSES)


if __name__ == "__main__":
    main()