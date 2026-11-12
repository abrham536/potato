import os
import random
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def generate_visualizations():
    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "raw" / "PlantVillage"
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_dir = Path(r"C:\Users\abreham\.gemini\antigravity\brain\6301ef43-bee2-47bf-8263-e660c8d0c4ad")

    # 1. Class Distribution Bar Chart
    classes = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    counts = [len([f for f in (raw_dir / c).glob("*") if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]) for c in classes]
    clean_labels = [c.replace("Potato___", "").replace("_", " ").title() for c in classes]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(clean_labels, counts, color=['#e74c3c', '#2ecc71', '#e67e22'], width=0.55)
    plt.title("Potato Leaf Disease Dataset - Class Distribution", fontsize=13, weight='bold', pad=15)
    plt.ylabel("Number of Images", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.ylim(0, max(counts) * 1.15)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 25, f"{count}",
                 ha='center', va='bottom', fontsize=11, weight='bold')

    plt.tight_layout()
    dist_path = figures_dir / "class_distribution.png"
    plt.savefig(dist_path, dpi=150)
    if artifact_dir.exists():
        plt.savefig(artifact_dir / "class_distribution.png", dpi=150)
    plt.close()
    print(f"Saved distribution plot to: {dist_path}")

    # 2. Sample Images Grid (3 per class)
    random.seed(42)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    plt.suptitle("Sample Images per Potato Health Category", fontsize=14, weight='bold', y=0.98)

    for row_idx, cls_name in enumerate(classes):
        img_paths = list((raw_dir / cls_name).glob("*.JPG")) + list((raw_dir / cls_name).glob("*.jpg"))
        sampled = random.sample(img_paths, 3)
        for col_idx, img_p in enumerate(sampled):
            ax = axes[row_idx, col_idx]
            with Image.open(img_p) as img:
                ax.imshow(img)
            ax.axis('off')
            if col_idx == 0:
                ax.set_title(clean_labels[row_idx], fontsize=12, weight='bold', loc='left')

    plt.tight_layout()
    samples_path = figures_dir / "sample_images.png"
    plt.savefig(samples_path, dpi=150)
    if artifact_dir.exists():
        plt.savefig(artifact_dir / "sample_images.png", dpi=150)
    plt.close()
    print(f"Saved sample grid plot to: {samples_path}")

if __name__ == "__main__":
    generate_visualizations()
