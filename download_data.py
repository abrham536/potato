import os
import shutil
import kagglehub
from pathlib import Path

def main():
    print("Downloading dataset 'aarishasifkhan/plantvillage-potato-disease-dataset' via kagglehub...")
    download_path = kagglehub.dataset_download("aarishasifkhan/plantvillage-potato-disease-dataset")
    print(f"Downloaded to cache: {download_path}")

    project_root = Path(__file__).resolve().parent
    dest_dir = project_root / "data" / "raw"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying dataset files to local project directory: {dest_dir} ...")
    # Inspect downloaded path contents
    src_path = Path(download_path)
    
    # If the download folder contains subfolders or directly class folders
    for item in src_path.iterdir():
        target = dest_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    print("Data download and local copy completed successfully!")

if __name__ == "__main__":
    main()
