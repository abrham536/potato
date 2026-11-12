import os
import random
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def demonstrate_augmentation():
    """
    Demonstrates data augmentation techniques visually by taking a single
    potato leaf image and applying diverse augmentations (flips, rotations,
    zooms, brightness adjustments) to generate a 3x3 augmented comparison.
    """
    project_root = Path(__file__).resolve().parent.parent
    train_dir = project_root / "data" / "processed" / "train"
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(r"C:\Users\abreham\.gemini\antigravity\brain\6301ef43-bee2-47bf-8263-e660c8d0c4ad")

    # Pick a sample Early Blight image
    sample_images = list((train_dir / "Potato___Early_blight").glob("*.JPG")) + list((train_dir / "Potato___Early_blight").glob("*.jpg"))
    if not sample_images:
        print("No sample images found. Ensure data is prepared.")
        return

    sample_img_path = sample_images[0]
    
    # We will import tensorflow if available, otherwise Pillow transforms as fallback
    try:
        import tensorflow as tf
        from tensorflow.keras import layers

        img = tf.keras.utils.load_img(sample_img_path, target_size=(256, 256))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # (1, 256, 256, 3)

        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.20),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ])

        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        plt.suptitle("Data Augmentation Pipeline - Transformations of Single Leaf", fontsize=13, weight='bold', y=0.98)

        for i, ax in enumerate(axes.flat):
            if i == 0:
                ax.imshow(img)
                ax.set_title("Original (256x256)", fontsize=11, weight='bold', color='navy')
            else:
                augmented_img = data_augmentation(img_array, training=True)
                ax.imshow(augmented_img[0].numpy().astype("uint8"))
                ax.set_title(f"Augmented Variation #{i}", fontsize=10)
            ax.axis('off')

    except ImportError:
        # Fallback using PIL
        with Image.open(sample_img_path).resize((256, 256)) as original:
            fig, axes = plt.subplots(3, 3, figsize=(9, 9))
            plt.suptitle("Data Augmentation Pipeline - PIL Fallback", fontsize=13, weight='bold', y=0.98)
            for i, ax in enumerate(axes.flat):
                if i == 0:
                    ax.imshow(original)
                    ax.set_title("Original (256x256)", fontsize=11, weight='bold', color='navy')
                else:
                    # Random rotation & flip
                    angle = random.choice([90, 180, 270, 15, -15, 30, -30])
                    variant = original.rotate(angle)
                    if random.random() > 0.5:
                        variant = variant.transpose(Image.FLIP_LEFT_RIGHT)
                    ax.imshow(variant)
                    ax.set_title(f"Augmented Variation #{i}", fontsize=10)
                ax.axis('off')

    plt.tight_layout()
    aug_path = figures_dir / "augmentation_samples.png"
    plt.savefig(aug_path, dpi=150)
    if artifact_dir.exists():
        plt.savefig(artifact_dir / "augmentation_samples.png", dpi=150)
    plt.close()
    print(f"Saved augmentation demonstration to: {aug_path}")

if __name__ == "__main__":
    demonstrate_augmentation()
