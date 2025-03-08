import os
import kaggle
import zipfile
import shutil
from pathlib import Path

def download_dataset():
    """
    Downloads the Chest X-Ray Images (Pneumonia) dataset from Kaggle.
    Requires Kaggle API credentials to be set up.
    """
    # Define paths
    data_dir = Path('../../data')
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    # Create directories if they don't exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Download dataset from Kaggle
    print("Downloading Chest X-Ray Images (Pneumonia) dataset from Kaggle...")
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'paultimothymooney/chest-xray-pneumonia',
            path=raw_dir,
            unzip=True
        )
        print("Dataset downloaded successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTo use the Kaggle API, you need to:")
        print("1. Create a Kaggle account: https://www.kaggle.com/")
        print("2. Go to your account settings and create an API token")
        print("3. Save the kaggle.json file to ~/.kaggle/kaggle.json")
        print("4. Ensure the file has permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Check if the dataset was downloaded correctly
    chest_xray_dir = raw_dir / 'chest_xray'
    if not chest_xray_dir.exists():
        print("Dataset structure is not as expected. Please check the downloaded files.")
        return False
    
    print("Dataset downloaded and extracted successfully!")
    return True

if __name__ == "__main__":
    download_dataset()