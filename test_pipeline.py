from pathlib import Path
from src.preprocess import build_data_pipeline

def test_pipeline():
    data_dir = Path("data") / "processed"
    pipeline = build_data_pipeline(data_dir, batch_size=32)
    
    print("Testing Train Batch loading...")
    for images, labels in pipeline["train"].take(1):
        print(f"Batch loaded successfully!")
        print(f" - Images shape: {images.shape} (batch_size, height, width, channels)")
        print(f" - Labels shape: {labels.shape} (batch_size, num_classes)")
        print(f" - Pixel range min: {images.numpy().min():.3f}, max: {images.numpy().max():.3f}")
        break

if __name__ == "__main__":
    test_pipeline()
