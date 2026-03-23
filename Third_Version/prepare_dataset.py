"""
Data Preparation Script for Driver Distraction Detection
Helps organize and prepare dataset for YOLOv8 training
"""

import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm
import json

def create_directory_structure(base_path):
    """
    Create YOLOv8 dataset directory structure
    
    Structure:
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
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        # Create image directories
        img_dir = os.path.join(base_path, 'images', split)
        os.makedirs(img_dir, exist_ok=True)
        
        # Create label directories
        lbl_dir = os.path.join(base_path, 'labels', split)
        os.makedirs(lbl_dir, exist_ok=True)
    
    print(f"✓ Directory structure created at: {base_path}")
    return base_path


def split_dataset(source_images, source_labels, output_path, 
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, 
                  seed=42):
    """
    Split dataset into train/val/test sets
    
    Args:
        source_images: Directory containing all images
        source_labels: Directory containing all label files
        output_path: Output directory for organized dataset
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
    """
    
    random.seed(seed)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(Path(source_images).glob(f'*{ext}'))
    
    print(f"Found {len(all_images)} images")
    
    # Shuffle
    random.shuffle(all_images)
    
    # Calculate split sizes
    total = len(all_images)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split data
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_images)} images ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_images)} images ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_images)} images ({test_ratio*100:.1f}%)")
    
    # Copy files to respective directories
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split_name, images in splits.items():
        print(f"\nCopying {split_name} set...")
        
        for img_path in tqdm(images):
            # Copy image
            img_dest = os.path.join(output_path, 'images', split_name, img_path.name)
            shutil.copy2(img_path, img_dest)
            
            # Copy corresponding label file
            label_name = img_path.stem + '.txt'
            label_path = os.path.join(source_labels, label_name)
            
            if os.path.exists(label_path):
                label_dest = os.path.join(output_path, 'labels', split_name, label_name)
                shutil.copy2(label_path, label_dest)
            else:
                print(f"Warning: Label not found for {img_path.name}")
    
    print("\n✓ Dataset split completed!")


def verify_dataset(dataset_path):
    """
    Verify dataset integrity and print statistics
    
    Args:
        dataset_path: Path to dataset root
    """
    
    print("\n" + "="*60)
    print("Dataset Verification")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        img_dir = os.path.join(dataset_path, 'images', split)
        lbl_dir = os.path.join(dataset_path, 'labels', split)
        
        images = list(Path(img_dir).glob('*.[jp][pn][g]'))
        labels = list(Path(lbl_dir).glob('*.txt'))
        
        print(f"\n{split.upper()} Set:")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")
        
        # Check for missing labels
        missing_labels = 0
        for img in images:
            label_path = os.path.join(lbl_dir, img.stem + '.txt')
            if not os.path.exists(label_path):
                missing_labels += 1
        
        if missing_labels > 0:
            print(f"  ⚠ Missing labels: {missing_labels}")
        else:
            print(f"  ✓ All images have labels")
        
        # Count class distribution
        if len(labels) > 0:
            class_counts = {}
            for label_file in labels:
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
            print(f"  Class distribution:")
            for class_id, count in sorted(class_counts.items()):
                print(f"    Class {class_id}: {count} instances")


def convert_csv_to_yolo(csv_file, output_label_dir, img_width=640, img_height=480):
    """
    Convert CSV annotations to YOLO format
    (Customize this based on your CSV format)
    
    YOLO format per line: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1]
    
    Args:
        csv_file: Path to CSV file with annotations
        output_label_dir: Directory to save YOLO format labels
        img_width: Image width
        img_height: Image height
    """
    
    import csv
    
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Example CSV format: image_name, class_id, x_min, y_min, x_max, y_max
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        current_image = None
        annotations = []
        
        for row in reader:
            image_name = row['image_name']
            
            # If new image, save previous annotations
            if current_image and current_image != image_name:
                label_path = os.path.join(output_label_dir, 
                                         Path(current_image).stem + '.txt')
                with open(label_path, 'w') as lf:
                    lf.write('\n'.join(annotations))
                annotations = []
            
            current_image = image_name
            
            # Convert to YOLO format
            class_id = int(row['class_id'])
            x_min = float(row['x_min'])
            y_min = float(row['y_min'])
            x_max = float(row['x_max'])
            y_max = float(row['y_max'])
            
            # Calculate normalized center and dimensions
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # Format: class_id x_center y_center width height
            annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            annotations.append(annotation)
        
        # Save last image annotations
        if annotations:
            label_path = os.path.join(output_label_dir, 
                                     Path(current_image).stem + '.txt')
            with open(label_path, 'w') as lf:
                lf.write('\n'.join(annotations))
    
    print(f"✓ Converted CSV to YOLO format: {output_label_dir}")


def main():
    """Main data preparation pipeline"""
    
    print("="*60)
    print("Driver Distraction Dataset Preparation")
    print("="*60)
    
    # Configuration
    OUTPUT_DATASET_PATH = './driver_distraction_dataset'
    
    # Option 1: If you have raw images and labels in separate directories
    # SOURCE_IMAGES = './raw_data/images'
    # SOURCE_LABELS = './raw_data/labels'
    
    # Step 1: Create directory structure
    print("\nStep 1: Creating directory structure...")
    create_directory_structure(OUTPUT_DATASET_PATH)
    
    # Step 2: Split dataset (uncomment if you have source data)
    # print("\nStep 2: Splitting dataset...")
    # split_dataset(
    #     source_images=SOURCE_IMAGES,
    #     source_labels=SOURCE_LABELS,
    #     output_path=OUTPUT_DATASET_PATH,
    #     train_ratio=0.7,
    #     val_ratio=0.2,
    #     test_ratio=0.1
    # )
    
    # Step 3: Verify dataset (uncomment after splitting)
    # print("\nStep 3: Verifying dataset...")
    # verify_dataset(OUTPUT_DATASET_PATH)
    
    print("\n" + "="*60)
    print("Data preparation setup completed!")
    print(f"Dataset structure created at: {OUTPUT_DATASET_PATH}")
    print("\nNext steps:")
    print("1. Place your images and labels in the appropriate directories")
    print("2. Run verify_dataset() to check integrity")
    print("3. Use train_distraction_model.py to train the model")
    print("="*60)


if __name__ == "__main__":
    main()
