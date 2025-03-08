import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

def preprocess_images(img_size=(224, 224)):
    """
    Preprocess the chest X-ray images for model training.
    - Resize images to a standard size
    - Convert to grayscale if needed
    - Normalize pixel values
    - Create train/val/test splits if not already done
    
    Args:
        img_size (tuple): Target image size (height, width)
    """
    # Define paths
    data_dir = Path('../../data')
    raw_dir = data_dir / 'raw' / 'chest_xray'
    processed_dir = data_dir / 'processed'
    
    # Check if raw data exists
    if not raw_dir.exists():
        print(f"Raw data directory not found at {raw_dir}")
        print("Please run download_data.py first.")
        return False
    
    # Create processed directories
    for split in ['train', 'val', 'test']:
        for label in ['NORMAL', 'PNEUMONIA']:
            os.makedirs(processed_dir / split / label, exist_ok=True)
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        split_dir = raw_dir / split
        if not split_dir.exists():
            print(f"Split directory {split_dir} not found")
            continue
        
        # Process each class (NORMAL, PNEUMONIA)
        for label in ['NORMAL', 'PNEUMONIA']:
            label_dir = split_dir / label
            if not label_dir.exists():
                print(f"Label directory {label_dir} not found")
                continue
            
            # Get all image files
            image_files = list(label_dir.glob('*.jpeg')) + list(label_dir.glob('*.jpg')) + list(label_dir.glob('*.png'))
            print(f"Processing {len(image_files)} {label} images in {split} set...")
            
            # Process each image
            for img_path in tqdm(image_files):
                try:
                    # Read image
                    img = cv2.imread(str(img_path))
                    
                    # Convert to grayscale if it's a color image
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Resize image
                    img = cv2.resize(img, img_size)
                    
                    # Normalize pixel values to [0, 1]
                    img = img / 255.0
                    
                    # Save processed image
                    output_path = processed_dir / split / label / img_path.name
                    cv2.imwrite(str(output_path), (img * 255).astype(np.uint8))
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Check if validation set exists, if not create it from training set
    val_dir = processed_dir / 'val'
    if not (val_dir / 'NORMAL').exists() or len(list((val_dir / 'NORMAL').glob('*'))) == 0:
        print("Validation set not found or empty. Creating validation set from training data...")
        create_validation_set(processed_dir)
    
    print("Preprocessing completed successfully!")
    return True

def create_validation_set(processed_dir, val_split=0.1):
    """
    Create a validation set from the training data if it doesn't exist.
    
    Args:
        processed_dir (Path): Path to the processed data directory
        val_split (float): Fraction of training data to use for validation
    """
    train_dir = processed_dir / 'train'
    val_dir = processed_dir / 'val'
    
    # Create validation directories
    for label in ['NORMAL', 'PNEUMONIA']:
        os.makedirs(val_dir / label, exist_ok=True)
        
        # Get all training images for this class
        train_images = list((train_dir / label).glob('*.jpeg')) + \
                      list((train_dir / label).glob('*.jpg')) + \
                      list((train_dir / label).glob('*.png'))
        
        # Determine number of images for validation
        num_val = int(len(train_images) * val_split)
        val_images = np.random.choice(train_images, num_val, replace=False)
        
        # Move selected images to validation directory
        for img_path in val_images:
            dest_path = val_dir / label / img_path.name
            shutil.move(str(img_path), str(dest_path))
        
        print(f"Moved {len(val_images)} {label} images to validation set")

if __name__ == "__main__":
    preprocess_images()