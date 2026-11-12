import os
from pathlib import Path
from PIL import Image

def get_pytorch_transforms(img_size=(256, 256)):
    """
    Returns standard PyTorch torchvision transforms for training and validation/testing.
    """
    try:
        from torchvision import transforms

        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(), # scales pixel values to [0.0, 1.0]
        ])

        val_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        return train_transform, val_transform
    except ImportError:
        return None, None

def get_pytorch_dataloaders(data_dir: Path, batch_size: int = 32):
    """
    Builds PyTorch DataLoaders for Train, Val, and Test splits.
    """
    try:
        import torch
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader

        train_transform, val_transform = get_pytorch_transforms()
        if train_transform is None:
            raise ImportError("torchvision is not installed.")

        train_dataset = ImageFolder(root=data_dir / "train", transform=train_transform)
        val_dataset = ImageFolder(root=data_dir / "val", transform=val_transform)
        test_dataset = ImageFolder(root=data_dir / "test", transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "classes": train_dataset.classes,
            "class_to_idx": train_dataset.class_to_idx
        }
    except Exception as e:
        print(f"PyTorch DataLoaders setup note: {e}")
        return None
