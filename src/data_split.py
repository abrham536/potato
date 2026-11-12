import os
import shutil
import random
from pathlib import Path

def create_dataset_splits(
    raw_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
):
    """
    Creates a stratified train / val / test split by copying files
    into output_dir/train, output_dir/val, output_dir/test.
    """
    assert round(train_ratio + val_ratio + test_ratio, 5) == 1.0, "Ratios must sum to 1.0"
    random.seed(seed)

    print("=" * 60)
    print("STRATIFIED DATASET SPLITTER")
    print("=" * 60)
    print(f"Source: {raw_dir.resolve()}")
    print(f"Target: {output_dir.resolve()}")
    print(f"Split ratios: {train_ratio*100:.0f}% Train / {val_ratio*100:.0f}% Val / {test_ratio*100:.0f}% Test")
    print(f"Random Seed: {seed}\n")

    classes = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not classes:
        raise ValueError(f"No class folders found in {raw_dir}")

    split_counts = {"train": {}, "val": {}, "test": {}}

    for cls in sorted(classes):
        cls_name = cls.name
        images = [f for f in cls.glob("*") if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        random.shuffle(images)

        total_cls = len(images)
        n_train = int(total_cls * train_ratio)
        n_val = int(total_cls * val_ratio)
        # Remaining goes to test to guarantee all images are used
        n_test = total_cls - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        print(f"Class '{cls_name}': Total {total_cls} -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

        for split_name, split_files in splits.items():
            dest_class_dir = output_dir / split_name / cls_name
            dest_class_dir.mkdir(parents=True, exist_ok=True)
            split_counts[split_name][cls_name] = len(split_files)

            for img_path in split_files:
                target_path = dest_class_dir / img_path.name
                shutil.copy2(img_path, target_path)

    print("\n" + "-" * 60)
    print("SPLIT SUMMARY:")
    for split_name in ["train", "val", "test"]:
        total_split = sum(split_counts[split_name].values())
        print(f" - {split_name.upper():5s}: {total_split:4d} images")
    print("-" * 60)

if __name__ == "__main__":
    raw_path = Path("data") / "raw" / "PlantVillage"
    processed_path = Path("data") / "processed"
    create_dataset_splits(raw_path, processed_path)
