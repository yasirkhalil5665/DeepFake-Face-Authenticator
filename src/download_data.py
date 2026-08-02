
import zipfile
from pathlib import Path


def download_kaggle_dataset(dataset: str, dest_dir: str = "data") -> Path:
   
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    import kaggle  

    kaggle.api.authenticate()

    zip_name = dataset.split("/")[-1] + ".zip"
    if not Path(zip_name).exists():
        print(f"[INFO] Downloading '{dataset}'...")
        kaggle.api.dataset_download_files(dataset, path=".", unzip=False)
    else:
        print(f"[INFO] '{zip_name}' already exists, skipping download.")

    print(f"[INFO] Extracting to '{dest_path}'...")
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(dest_path)

    print(f"[INFO] Done. Data available at: {dest_path}")
    return dest_path

if __name__ == "__main__":
    download_kaggle_dataset("ajaysonicu/deepfakefusion-399k-realfake-faces")
