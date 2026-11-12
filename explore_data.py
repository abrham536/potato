import os
from pathlib import Path
from PIL import Image

def analyze_dataset(data_dir: Path):
    print("=" * 60)
    print("DATASET EXPLORATION & AUDIT")
    print("=" * 60)
    print(f"Dataset root: {data_dir.resolve()}\n")

    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist.")
        return

    classes = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes:")
    
    total_images = 0
    class_counts = {}
    sample_sizes = set()
    sample_modes = set()

    for cls in sorted(classes):
        image_files = [f for f in cls.glob("*") if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        count = len(image_files)
        class_counts[cls.name] = count
        total_images += count
        print(f" - {cls.name:30s}: {count:5d} images")

        # Check resolution and mode of first 10 images in each class
        for img_path in image_files[:10]:
            try:
                with Image.open(img_path) as img:
                    sample_sizes.add(img.size) # (width, height)
                    sample_modes.add(img.mode)
            except Exception as e:
                print(f"   [!] Corrupted image detected: {img_path} ({e})")

    print("\n" + "-" * 60)
    print(f"Total images across all classes: {total_images}")
    print(f"Image resolutions found in sample: {sample_sizes}")
    print(f"Color modes found in sample: {sample_modes}")
    print("-" * 60)

if __name__ == "__main__":
    raw_dir = Path("data") / "raw" / "PlantVillage"
    analyze_dataset(raw_dir)
