"""
Dataset Download and Preparation Helper
Downloads State Farm Distracted Driver Dataset from Kaggle and prepares it for YOLOv8
"""

import os
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm
import random
import json

def setup_kaggle():
    """
    Setup Kaggle API credentials
    """
    print("\n" + "="*60)
    print("Kaggle API Setup")
    print("="*60)
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print("✓ Kaggle credentials found")
        return True
    
    print("\n⚠ Kaggle credentials not found!")
    print("\nTo download from Kaggle, you need API credentials:")
    print("1. Go to https://www.kaggle.com/")
    print("2. Click on your profile picture → Account")
    print("3. Scroll to 'API' section → Click 'Create New API Token'")
    print("4. This downloads kaggle.json")
    print(f"5. Move kaggle.json to: {kaggle_dir}/")
    print("\nOr manually run:")
    print(f"  mkdir -p {kaggle_dir}")
    print(f"  mv ~/Downloads/kaggle.json {kaggle_dir}/")
    print(f"  chmod 600 {kaggle_dir}/kaggle.json")
    
    return False


def download_state_farm_dataset(output_dir='raw_data'):
    """
    Download State Farm dataset from Kaggle
    """
    
    if not setup_kaggle():
        return False
    
    print("\n" + "="*60)
    print("Downloading State Farm Dataset")
    print("="*60)
    
    try:
        import kaggle
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download dataset
        print("\nDownloading... (this may take several minutes)")
        kaggle.api.competition_download_files(
            'state-farm-distracted-driver-detection',
            path=output_dir
        )
        
        # Extract files
        zip_path = os.path.join(output_dir, 'state-farm-distracted-driver-detection.zip')
        if os.path.exists(zip_path):
            print("\nExtracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Remove zip file
            os.remove(zip_path)
            print("✓ Dataset downloaded and extracted")
            return True
        
    except ImportError:
        print("\n❌ Kaggle package not installed!")
        print("Install it with: pip install kaggle")
        return False
    
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {str(e)}")
        return False
    
    return True


def convert_state_farm_to_yolo(source_dir='raw_data', output_dir='driver_distraction_dataset'):
    """
    Convert State Farm dataset structure to YOLO format
    
    State Farm structure:
    imgs/
    ├── train/
    │   ├── c0/  (safe driving)
    │   ├── c1/  (texting right)
    │   └── ...
    └── test/
    
    YOLO structure:
    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
    """
    
    print("\n" + "="*60)
    print("Converting to YOLO Format")
    print("="*60)
    
    # Define class mapping
    class_mapping = {
        'c0': 0,  # safe_driving
        'c1': 1,  # texting_right
        'c2': 2,  # talking_phone_right
        'c3': 3,  # texting_left
        'c4': 4,  # talking_phone_left
        'c5': 5,  # adjusting_radio
        'c6': 6,  # drinking
        'c7': 7,  # reaching_behind
        'c8': 8,  # hair_makeup
        'c9': 9   # talking_passenger
    }
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # Process training data
    train_dir = os.path.join(source_dir, 'imgs', 'train')
    
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        return False
    
    print("\nProcessing training images...")
    
    all_images = []
    
    # Collect all images with their class labels
    for class_folder in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_folder)
        
        if not os.path.isdir(class_path):
            continue
        
        if class_folder not in class_mapping:
            continue
        
        class_id = class_mapping[class_folder]
        
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append({
                    'path': os.path.join(class_path, img_name),
                    'class': class_id,
                    'name': img_name
                })
    
    print(f"Found {len(all_images)} training images")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(all_images)
    
    # 70% train, 20% val, 10% test
    train_size = int(len(all_images) * 0.7)
    val_size = int(len(all_images) * 0.2)
    
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val:   {len(val_images)} images")
    print(f"  Test:  {len(test_images)} images")
    
    # Process each split
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split_name, images in splits.items():
        print(f"\nProcessing {split_name} set...")
        
        for img_data in tqdm(images):
            # Copy image
            img_dest = os.path.join(output_dir, 'images', split_name, img_data['name'])
            shutil.copy2(img_data['path'], img_dest)
            
            # Create label file
            # For classification, we create a simple bounding box covering the whole image
            # You can adjust this if you have actual bounding box annotations
            label_name = Path(img_data['name']).stem + '.txt'
            label_path = os.path.join(output_dir, 'labels', split_name, label_name)
            
            # YOLO format: class_id x_center y_center width height (normalized)
            # Using full image as bounding box
            with open(label_path, 'w') as f:
                f.write(f"{img_data['class']} 0.5 0.5 0.9 0.9\n")
    
    print("\n✓ Conversion completed!")
    
    # Create dataset statistics
    stats = {
        'total_images': len(all_images),
        'train': len(train_images),
        'val': len(val_images),
        'test': len(test_images),
        'classes': class_mapping
    }
    
    stats_path = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset statistics saved to: {stats_path}")
    
    return True


def create_dataset_yaml(output_dir='driver_distraction_dataset'):
    """
    Create dataset.yaml for YOLOv8 training
    """
    
    yaml_content = f"""# Driver Distraction Dataset Configuration
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

# Classes
nc: 10
names:
  0: safe_driving
  1: texting_right
  2: talking_phone_right
  3: texting_left
  4: talking_phone_left
  5: adjusting_radio
  6: drinking
  7: reaching_behind
  8: hair_makeup
  9: talking_passenger
"""
    
    yaml_path = 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Dataset YAML created: {yaml_path}")
    return yaml_path


def main():
    """
    Main pipeline to download and prepare dataset
    """
    
    print("="*60)
    print("State Farm Dataset Downloader and Converter")
    print("="*60)
    
    # Step 1: Download dataset
    print("\n[Step 1/3] Downloading dataset...")
    if not download_state_farm_dataset():
        print("\n❌ Dataset download failed!")
        print("\nAlternative: Manually download the dataset from:")
        print("https://www.kaggle.com/c/state-farm-distracted-driver-detection/data")
        print("Then extract to 'raw_data/' directory")
        return
    
    # Step 2: Convert to YOLO format
    print("\n[Step 2/3] Converting to YOLO format...")
    if not convert_state_farm_to_yolo():
        print("\n❌ Conversion failed!")
        return
    
    # Step 3: Create dataset YAML
    print("\n[Step 3/3] Creating dataset configuration...")
    create_dataset_yaml()
    
    print("\n" + "="*60)
    print("✓ Dataset preparation completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify dataset structure:")
    print("   python prepare_dataset.py")
    print("\n2. Start training:")
    print("   python train_distraction_model.py")
    print("\n3. Test the model:")
    print("   python test_model.py --source webcam")
    print("="*60)


if __name__ == "__main__":
    main()
