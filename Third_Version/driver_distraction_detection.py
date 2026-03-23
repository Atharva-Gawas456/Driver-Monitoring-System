"""
Driver Distraction Detection System using YOLOv8
Monitors driver behavior and detects distracted driving activities
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
from datetime import datetime
import os

class DriverDistractionDetector:
    def __init__(self, model_path='best.pt', confidence_threshold=0.5):
        """
        Initialize the Driver Distraction Detector
        
        Args:
            model_path: Path to trained YOLOv8 model
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Define distraction classes (customize based on your dataset)
        self.distraction_classes = {
            'safe_driving': 0,
            'texting_right': 1,
            'talking_phone_right': 2,
            'texting_left': 3,
            'talking_phone_left': 4,
            'adjusting_radio': 5,
            'drinking': 6,
            'reaching_behind': 7,
            'hair_makeup': 8,
            'talking_passenger': 9
        }
        
        # Reverse mapping for display
        self.class_names = {v: k for k, v in self.distraction_classes.items()}
        
        # Alert system
        self.alert_history = deque(maxlen=30)  # Last 30 frames
        self.distraction_count = 0
        self.alert_threshold = 15  # Alert if distracted for 15 consecutive frames
        
        # Statistics
        self.total_frames = 0
        self.distracted_frames = 0
        
        # Colors for different alerts
        self.colors = {
            'safe': (0, 255, 0),      # Green
            'warning': (0, 165, 255),  # Orange
            'danger': (0, 0, 255)      # Red
        }
        
    def is_distracted(self, class_id):
        """Check if detected class indicates distraction"""
        return class_id != self.distraction_classes.get('safe_driving', 0)
    
    def get_alert_level(self):
        """Determine alert level based on recent detections"""
        if len(self.alert_history) < 10:
            return 'safe'
        
        distracted_count = sum(self.alert_history)
        ratio = distracted_count / len(self.alert_history)
        
        if ratio > 0.7:
            return 'danger'
        elif ratio > 0.3:
            return 'warning'
        else:
            return 'safe'
    
    def draw_alert_panel(self, frame, alert_level, current_action):
        """Draw alert panel on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Draw alert bar at top
        bar_height = 80
        cv2.rectangle(overlay, (0, 0), (width, bar_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Alert status
        color = self.colors[alert_level]
        status_text = alert_level.upper()
        cv2.putText(frame, status_text, (20, 35), 
                   cv2.FONT_HERSHEY_BOLD, 1.2, color, 3)
        
        # Current action
        action_text = f"Action: {current_action.replace('_', ' ').title()}"
        cv2.putText(frame, action_text, (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Statistics panel
        stats_y = bar_height + 30
        distraction_rate = (self.distracted_frames / max(1, self.total_frames)) * 100
        
        cv2.putText(frame, f"Distraction Rate: {distraction_rate:.1f}%", 
                   (width - 350, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Warning message if distracted
        if alert_level in ['warning', 'danger']:
            warning_text = "⚠ DRIVER DISTRACTED - FOCUS ON ROAD!"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_BOLD, 1.0, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 50
            
            # Blinking effect for danger
            if alert_level == 'danger' and int(time.time() * 2) % 2 == 0:
                cv2.putText(frame, warning_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_BOLD, 1.0, self.colors['danger'], 3)
        
        return frame
    
    def detect_distraction(self, frame):
        """
        Detect driver distraction in frame
        
        Args:
            frame: Input video frame
            
        Returns:
            annotated_frame: Frame with detections and alerts
            is_distracted: Boolean indicating if driver is distracted
            confidence: Detection confidence
        """
        self.total_frames += 1
        
        # Run YOLOv8 inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Process results
        is_distracted = False
        max_confidence = 0
        detected_action = "unknown"
        
        for result in results:
            boxes = result.boxes
            
            if len(boxes) > 0:
                # Get highest confidence detection
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                
                max_idx = np.argmax(confidences)
                max_confidence = confidences[max_idx]
                detected_class = classes[max_idx]
                
                detected_action = self.class_names.get(detected_class, "unknown")
                is_distracted = self.is_distracted(detected_class)
                
                # Draw bounding boxes
                for box, conf, cls in zip(boxes.xyxy.cpu().numpy(), confidences, classes):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{self.class_names.get(cls, 'unknown')}: {conf:.2f}"
                    
                    # Color based on distraction
                    color = self.colors['danger'] if self.is_distracted(cls) else self.colors['safe']
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Update alert history
        self.alert_history.append(1 if is_distracted else 0)
        if is_distracted:
            self.distracted_frames += 1
        
        # Get alert level
        alert_level = self.get_alert_level()
        
        # Draw alert panel
        annotated_frame = self.draw_alert_panel(frame, alert_level, detected_action)
        
        return annotated_frame, is_distracted, max_confidence
    
    def process_video(self, video_source=0, output_path=None, display=True):
        """
        Process video stream for distraction detection
        
        Args:
            video_source: Video file path or camera index (0 for webcam)
            output_path: Path to save output video (optional)
            display: Whether to display video window
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Starting driver distraction detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect distraction
                annotated_frame, is_distracted, confidence = self.detect_distraction(frame)
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time
                
                # Draw FPS
                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                           (width - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write to output if specified
                if out:
                    out.write(annotated_frame)
                
                # Display
                if display:
                    cv2.imshow('Driver Distraction Detection', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(screenshot_path, annotated_frame)
                        print(f"Screenshot saved: {screenshot_path}")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
            
            # Print statistics
            print("\n=== Detection Summary ===")
            print(f"Total frames processed: {self.total_frames}")
            print(f"Distracted frames: {self.distracted_frames}")
            print(f"Distraction rate: {(self.distracted_frames/max(1, self.total_frames)*100):.2f}%")
            print(f"Average FPS: {current_fps:.2f}")


def main():
    """Main function to run the detector"""
    
    # Configuration
    MODEL_PATH = 'best.pt'  # Path to your trained YOLOv8 model
    VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
    OUTPUT_VIDEO = 'output_detection.mp4'  # Set to None to disable saving
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        print("Please train your YOLOv8 model first or specify correct path.")
        print("See the training script for details on how to train the model.")
        return
    
    # Initialize detector
    detector = DriverDistractionDetector(
        model_path=MODEL_PATH,
        confidence_threshold=0.5
    )
    
    # Process video
    detector.process_video(
        video_source=VIDEO_SOURCE,
        output_path=OUTPUT_VIDEO,
        display=True
    )


if __name__ == "__main__":
    main()
