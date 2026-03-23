# Driver Distraction Detection using YOLOv8

A comprehensive system for detecting distracted driving behaviors using YOLOv8 object detection. This system can identify various types of driver distractions in real-time from video streams or camera feeds.

## 🎯 Features

- **Real-time Detection**: Monitor driver behavior in real-time
- **Multiple Distraction Classes**: Detect 10 different distraction types
- **Alert System**: Visual warnings with three-level alert system (Safe, Warning, Danger)
- **Statistics Tracking**: Monitor distraction rates and patterns
- **Video Processing**: Process live camera feeds or recorded videos
- **Model Export**: Export to ONNX and other formats for deployment

## 📋 Detected Distraction Classes

1. **Safe Driving** - Normal, attentive driving
2. **Texting (Right Hand)** - Using phone to text with right hand
3. **Talking on Phone (Right Hand)** - Phone call with right hand
4. **Texting (Left Hand)** - Using phone to text with left hand
5. **Talking on Phone (Left Hand)** - Phone call with left hand
6. **Adjusting Radio** - Operating radio/entertainment controls
7. **Drinking** - Consuming beverages while driving
8. **Reaching Behind** - Reaching for objects in back seat
9. **Hair/Makeup** - Grooming activities
10. **Talking to Passenger** - Engaged conversation with passenger

## 🗂️ Recommended Datasets

### 1. State Farm Distracted Driver Detection Dataset ⭐ HIGHLY RECOMMENDED

**Source**: Kaggle Competition Dataset  
**Link**: https://www.kaggle.com/c/state-farm-distracted-driver-detection/data

**Details**:
- **Size**: ~22,424 images
- **Classes**: 10 classes (matches our model exactly)
- **Resolution**: ~640x480 pixels
- **Quality**: High-quality labeled images
- **Split**: Pre-split into train and test sets
- **Format**: Images with class labels

**Classes**:
- c0: Safe driving
- c1: Texting - right
- c2: Talking on phone - right
- c3: Texting - left
- c4: Talking on phone - left
- c5: Operating radio
- c6: Drinking
- c7: Reaching behind
- c8: Hair and makeup
- c9: Talking to passenger

**Why This Dataset?**
✅ Industry-standard dataset  
✅ Well-balanced classes  
✅ High-quality annotations  
✅ Large enough for good generalization  
✅ Widely used in research

### 2. AUC Distracted Driver Dataset

**Source**: American University in Cairo  
**Link**: https://abouelnaga.io/projects/auc-distracted-driver-dataset/

**Details**:
- **Size**: ~17,000+ images
- **Classes**: 10 classes
- **Features**: Multiple drivers, various lighting conditions
- **Quality**: Research-grade annotations

### 3. Custom Dataset Creation

If you want to create your own dataset:

**Tools**:
- **LabelImg**: https://github.com/tzutalin/labelImg
- **Roboflow**: https://roboflow.com/ (recommended for automatic formatting)
- **CVAT**: https://github.com/opencv/cvat

**Capture Guidelines**:
- Record from driver's seat perspective
- Capture various lighting conditions
- Include multiple subjects for diversity
- Ensure consistent image quality
- Aim for 500-1000 images per class minimum

### 4. Synthetic Data Augmentation

Use augmentation to expand your dataset:
- Rotation (±10°)
- Brightness/contrast adjustment
- Gaussian noise
- Horizontal flips
- Zoom/scale variations

## 📦 Installation

### Prerequisites

```bash
# Python 3.8 or higher required
python --version
```

### Install Dependencies

```bash
# Install PyTorch (visit pytorch.org for your specific CUDA version)
pip install torch torchvision

# Install Ultralytics YOLOv8
pip install ultralytics

# Install additional requirements
pip install opencv-python numpy pyyaml tqdm
```

## 🚀 Quick Start

### Step 1: Download Dataset

```bash
# Download State Farm Dataset from Kaggle
# You'll need Kaggle API credentials

# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle competitions download -c state-farm-distracted-driver-detection

# Unzip
unzip state-farm-distracted-driver-detection.zip -d raw_data/
```

### Step 2: Prepare Dataset

```bash
# Run the data preparation script
python prepare_dataset.py
```

**Manual Preparation**:

If you're using the State Farm dataset, you need to convert it to YOLO format:

1. **Organize images** by class into the structure:
```
driver_distraction_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

2. **Create YOLO labels** (format: one .txt file per image):
```
# Each line: <class_id> <x_center> <y_center> <width> <height>
# All values normalized to [0, 1]
0 0.5 0.5 0.8 0.8
```

**Using Roboflow (Easiest Method)**:
1. Upload images to Roboflow
2. Annotate bounding boxes
3. Export in "YOLOv8" format
4. Download and extract to `driver_distraction_dataset/`

### Step 3: Train the Model

```bash
# Start training
python train_distraction_model.py
```

**Training Parameters** (edit in script):
- `MODEL_SIZE`: 'n' (nano), 's', 'm', 'l', 'x' - larger = more accurate but slower
- `EPOCHS`: 100 (increase for better accuracy)
- `BATCH_SIZE`: 16 (adjust based on GPU memory)
- `IMAGE_SIZE`: 640 (standard for YOLOv8)

**Training Time Estimates**:
- **YOLOv8n** (~20K images): ~2-3 hours on RTX 3060
- **YOLOv8s** (~20K images): ~4-5 hours on RTX 3060
- **YOLOv8m** (~20K images): ~8-10 hours on RTX 3060

### Step 4: Run Detection

```bash
# On webcam
python driver_distraction_detection.py

# On video file
# Edit VIDEO_SOURCE in script to point to your video
python driver_distraction_detection.py
```

## 📊 Model Performance Expectations

With the State Farm dataset (after 100 epochs):

| Model | mAP50 | mAP50-95 | Speed (FPS) | Size |
|-------|-------|----------|-------------|------|
| YOLOv8n | ~0.92 | ~0.75 | 45-60 | 6 MB |
| YOLOv8s | ~0.94 | ~0.78 | 35-45 | 22 MB |
| YOLOv8m | ~0.96 | ~0.82 | 25-35 | 52 MB |

## 🎮 Usage Examples

### Basic Detection

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('best.pt')

# Run on image
results = model('driver_image.jpg')

# Display results
results[0].show()
```

### Real-time Webcam

```python
from driver_distraction_detection import DriverDistractionDetector

detector = DriverDistractionDetector(
    model_path='best.pt',
    confidence_threshold=0.5
)

detector.process_video(
    video_source=0,  # 0 for webcam
    display=True
)
```

### Process Video File

```python
detector = DriverDistractionDetector(model_path='best.pt')

detector.process_video(
    video_source='dashcam_video.mp4',
    output_path='annotated_output.mp4',
    display=True
)
```

## 🔧 Configuration

### Adjust Detection Sensitivity

In `driver_distraction_detection.py`:

```python
detector = DriverDistractionDetector(
    model_path='best.pt',
    confidence_threshold=0.3  # Lower = more sensitive (0.1-0.9)
)
```

### Modify Alert Thresholds

```python
# In DriverDistractionDetector class
self.alert_threshold = 15  # Frames before alert (lower = faster alerts)
```

### Change Alert Colors

```python
self.colors = {
    'safe': (0, 255, 0),      # Green
    'warning': (0, 165, 255),  # Orange
    'danger': (0, 0, 255)      # Red
}
```

## 📈 Improving Model Performance

### 1. Data Augmentation

Enable in training script:

```python
# Automatic augmentation is enabled by default in YOLOv8
# Adjust parameters:
hsv_h=0.015,      # Hue augmentation
hsv_s=0.7,        # Saturation
hsv_v=0.4,        # Value
degrees=0.0,      # Rotation
translate=0.1,    # Translation
scale=0.5,        # Scaling
fliplr=0.5,       # Horizontal flip
```

### 2. Increase Training Data

- Collect more diverse scenarios
- Include various lighting conditions
- Add different vehicle interiors
- Mix different demographics

### 3. Fine-tuning

```python
# Load your trained model
model = YOLO('best.pt')

# Continue training with new data
model.train(
    data='dataset.yaml',
    epochs=50,
    resume=True  # Resume from checkpoint
)
```

### 4. Hyperparameter Tuning

```python
# Adjust learning rate
model.train(
    lr0=0.01,      # Initial learning rate
    lrf=0.001,     # Final learning rate
    momentum=0.937,
    weight_decay=0.0005
)
```

## 🚨 Alert System Explained

### Safe (Green)
- Driver is attentive
- No distractions detected
- Normal driving behavior

### Warning (Orange)
- Occasional distraction detected
- 30-70% of recent frames show distraction
- Driver should refocus

### Danger (Red)
- Persistent distraction
- >70% of recent frames show distraction
- Immediate attention required
- Blinking warning message

## 📱 Deployment Options

### 1. Edge Device (Raspberry Pi, Jetson Nano)

```bash
# Export to optimized format
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='engine')"
```

### 2. Mobile (Android/iOS)

```bash
# Export to TFLite
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='tflite')"
```

### 3. Web Application

```bash
# Export to ONNX
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx')"
```

### 4. Cloud API

Deploy using:
- AWS Lambda with SageMaker
- Google Cloud Functions
- Azure ML
- Hugging Face Spaces

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size
```python
batch_size = 8  # Instead of 16
```

### Issue: Low Detection Accuracy

**Solutions**:
1. Train for more epochs (150-200)
2. Use larger model (YOLOv8m or YOLOv8l)
3. Increase dataset size
4. Balance class distribution
5. Clean noisy labels

### Issue: Slow FPS

**Solutions**:
1. Use smaller model (YOLOv8n)
2. Reduce image size: `imgsz=416`
3. Use GPU acceleration
4. Export to TensorRT for inference

### Issue: False Positives

**Solutions**:
1. Increase confidence threshold
2. Train with more negative examples
3. Use larger model for better discrimination

## 📚 Additional Resources

### YOLOv8 Documentation
- https://docs.ultralytics.com/

### Research Papers
- "Distracted Driver Detection using Deep Learning" (2019)
- "Real-Time Driver Distraction Detection" (2020)

### Related Projects
- Driver Drowsiness Detection
- Seatbelt Detection
- Phone Usage Detection

## 📄 License

This project is for educational and research purposes. Please check dataset licenses before commercial use.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional distraction classes
- Multi-face detection support
- Temporal analysis (action recognition)
- Integration with vehicle systems

## ⚠️ Disclaimer

This system is a research/educational tool and should not be solely relied upon for driver safety. Always practice safe driving habits.

## 🙏 Acknowledgments

- State Farm for the distracted driver dataset
- Ultralytics for YOLOv8
- OpenCV community

## 📧 Support

For issues and questions:
1. Check troubleshooting section
2. Review YOLOv8 documentation
3. Open an issue with detailed description

---

**Happy Coding! Drive Safe! 🚗💨**
