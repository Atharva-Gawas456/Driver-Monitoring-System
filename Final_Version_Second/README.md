# 🚗 Driver Drowsiness Detection

A deep learning project designed to detect driver drowsiness in real-time using a computer webcam. This system features continuous monitoring and an automated alert system to enhance road safety.

---

## 📝 Overview

### Execution Order

1.  **Environment Setup**
    Ensure you have Python installed, then install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Preparation**
    The model is trained using a specialized dataset from Roboflow. You can view or download it here:
    [Drowsiness Detection Dataset](https://universe.roboflow.com/karthik-madhvan/drowsiness-detection-xsriz)

    To retrain the model, use the following YOLOv8 command:
    ```bash
    yolo task=detect mode=train model=yolov8s.pt data={./data}/data.yaml epochs=100 imgsz=640
    ```

3.  **Execution**
    If you wish to skip training, use the provided `best.pt` file. Execute the application within your virtual environment:
    ```bash
    python main.py
    ```

---

### 📄 References
The algorithm and logic for this detection system follow the methodology detailed in this research paper:
[IJEAT Research Paper Link](https://www.ijeat.org/wp-content/uploads/papers/v8i5S/E10150585S19.pdf)

---

### 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-44A833?style=flat&logo=Anaconda&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=OpenCV&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00599C?style=flat&logo=Ultralytics&logoColor=white)

---

### 👥 Contributors

| Name | GitHub Profile |
| :--- | :--- |
| **Atharva Gawas** | [Atharva-Gawas456](https://github.com/Atharva-Gawas456) |

---

### 📁 Directory Structure
```text
driver-drowsiness-detection/
├── main.py                # Core application script
├── alarm.wav              # Audio alert file
├── best.pt                # Pre-trained YOLOv8 weights
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
