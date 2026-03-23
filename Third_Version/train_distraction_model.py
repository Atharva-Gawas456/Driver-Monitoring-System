"""
YOLOv8 Training Script for Driver Distraction Detection
Trains a model to detect various distracted driving behaviors
"""

from ultralytics import YOLO
import yaml
import os
from pathlib import Path

def create_dataset_yaml(dataset_path, output_path='dataset.yaml'):
    """
    Create YAML configuration file for YOLOv8 training
    
    Args:
        dataset_path: Root path to dataset
        output_path: Path to save YAML file
    """
    
    # Define distraction classes
    classes = {
        0: 'safe_driving',
        1: 'texting_right',
        2: 'talking_phone_right',
        3: 'texting_left',
        4: 'talking_phone_left',
        5: 'adjusting_radio',
        6: 'drinking',
        7: 'reaching_behind',
        8: 'hair_makeup',
        9: 'talking_passenger'
    }
    
    # Create YAML structure
    dataset_config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    # Write YAML file
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Dataset YAML created: {output_path}")
    return output_path


def train_model(dataset_yaml='dataset.yaml', 
                model_size='n',
                epochs=100, 
                imgsz=640,
                batch_size=16,
                device='0',
                patience=50):
    """
    Train YOLOv8 model for driver distraction detection
    
    Args:
        dataset_yaml: Path to dataset YAML file
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        imgsz: Image size for training
        batch_size: Batch size
        device: Device to use ('0' for GPU, 'cpu' for CPU)
        patience: Early stopping patience
    """
    
    print("="*60)
    print("YOLOv8 Driver Distraction Detection - Training")
    print("="*60)
    
    # Initialize model
    model_name = f'yolov8{model_size}.pt'
    print(f"\nInitializing YOLOv8 model: {model_name}")
    model = YOLO(model_name)
    
    # Training configuration
    print(f"\nTraining Configuration:")
    print(f"  - Dataset: {dataset_yaml}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Image Size: {imgsz}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Device: {device}")
    print(f"  - Patience: {patience}")
    
    # Train the model
    print("\nStarting training...")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        patience=patience,
        save=True,
        project='driver_distraction_runs',
        name='train',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    return model, results


def validate_model(model_path='driver_distraction_runs/train/weights/best.pt',
                   dataset_yaml='dataset.yaml',
                   imgsz=640):
    """
    Validate trained model
    
    Args:
        model_path: Path to trained model weights
        dataset_yaml: Path to dataset YAML
        imgsz: Image size for validation
    """
    
    print("\n" + "="*60)
    print("Model Validation")
    print("="*60)
    
    model = YOLO(model_path)
    
    # Validate
    metrics = model.val(
        data=dataset_yaml,
        imgsz=imgsz,
        batch=16,
        conf=0.25,
        iou=0.6,
        device='0'
    )
    
    # Print metrics
    print("\nValidation Metrics:")
    print(f"  - mAP50: {metrics.box.map50:.4f}")
    print(f"  - mAP50-95: {metrics.box.map:.4f}")
    print(f"  - Precision: {metrics.box.mp:.4f}")
    print(f"  - Recall: {metrics.box.mr:.4f}")
    
    return metrics


def export_model(model_path='driver_distraction_runs/train/weights/best.pt',
                 export_formats=['onnx', 'torchscript']):
    """
    Export model to different formats for deployment
    
    Args:
        model_path: Path to trained model
        export_formats: List of formats to export to
    """
    
    print("\n" + "="*60)
    print("Model Export")
    print("="*60)
    
    model = YOLO(model_path)
    
    for fmt in export_formats:
        print(f"\nExporting to {fmt.upper()}...")
        try:
            model.export(format=fmt, imgsz=640)
            print(f"✓ {fmt.upper()} export successful")
        except Exception as e:
            print(f"✗ {fmt.upper()} export failed: {str(e)}")


def main():
    """Main training pipeline"""
    
    # ============================================
    # CONFIGURATION
    # ============================================
    
    # Dataset path - UPDATE THIS to your dataset location
    DATASET_PATH = './driver_distraction_dataset'
    
    # Training parameters
    MODEL_SIZE = 'n'  # 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
    EPOCHS = 100
    IMAGE_SIZE = 640
    BATCH_SIZE = 16
    DEVICE = '0'  # '0' for GPU, 'cpu' for CPU
    
    # ============================================
    # TRAINING PIPELINE
    # ============================================
    
    # Step 1: Create dataset YAML
    print("Step 1: Creating dataset configuration...")
    dataset_yaml = create_dataset_yaml(DATASET_PATH)
    
    # Step 2: Train model
    print("\nStep 2: Training model...")
    model, results = train_model(
        dataset_yaml=dataset_yaml,
        model_size=MODEL_SIZE,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        patience=50
    )
    
    # Step 3: Validate model
    print("\nStep 3: Validating model...")
    best_model_path = 'driver_distraction_runs/train/weights/best.pt'
    metrics = validate_model(best_model_path, dataset_yaml, IMAGE_SIZE)
    
    # Step 4: Export model (optional)
    print("\nStep 4: Exporting model...")
    export_model(best_model_path, export_formats=['onnx'])
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print(f"Best model saved at: {best_model_path}")
    print("="*60)


if __name__ == "__main__":
    main()
