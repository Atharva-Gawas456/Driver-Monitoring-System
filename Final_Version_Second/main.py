'''

1. Implementation completed on laptop environment
2. Model name (best.pt)
3. requirements.txt modified (added pygame package)
4. Separate queues created for drowsiness, yawning, and head movement (BB colors: Red, Yellow, Yellow)
5. Strong drowsiness + weak drowsiness → merged into "drowsy" (800) / average human yawn duration (6000) → (testing 1000) / head tilt up/down (800)
6. Warning message displayed on screen when drowsiness, yawning, or head movement is detected under specific conditions
7. Added alarm file alarm.wav
8. Drowsiness detection → alarm repeats until user exits drowsy state (3 seconds)
9. Yawning, head movement detection → alarm plays once (1 second) and resets
'''

import cv2
import torch
import pygame
import numpy as np
from ultralytics import YOLO
from collections import deque
import datetime

# Define constants
FPS = 30  # Frame rate
WARNING_DURATION = 2  # Warning display duration (seconds)
QUEUE_DURATION = 2  # Duration to store in queue (seconds)
YAWN_THRESHOLD_FRAMES = int(FPS * 1)  # Yawn frame threshold -> set to ~1 sec for demo
# DROWSY_THRESHOLD_FRAMES = int(FPS * 0.4)  # Weak drowsiness threshold -> removed
DROWSY_THRESHOLD_FRAMES = int(FPS * 0.8)  # Strong drowsiness threshold -> all sleep → drowsy
HEAD_THRESHOLD_FRAMES = int(FPS * 0.8)  # Head movement threshold

def play_alarm(sound_file, duration):
    # Initialize pygame mixer
    pygame.mixer.init()
    # Load sound file
    alarm_sound = pygame.mixer.Sound(sound_file)
    # Play sound (for specified duration)
    alarm_sound.play(loops=0, maxtime=duration)  # Loop used to repeatedly play clipped alarm from an 8-second file

def trigger_alarm(trigger, sound_file, duration):
    if trigger:
        print("Alarm is ringing!")
        play_alarm(sound_file, duration)
    else:
        print("Alarm is not triggered.")

def get_webcam_fps():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Cannot access webcam.")
        return None
    
    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30

def load_model(model_path):
    # Load YOLOv8 model
    model = YOLO(model_path)
    return model

def webcam_detection(model, fps):
    queue_length = int(fps * QUEUE_DURATION)
    # Redefinition
    # drowsy_threshold_frames = int(fps * 0.4)  # Weak drowsiness -> removed
    drowsy_threshold_frames = int(fps * 0.8)  # Strong drowsiness -> drowsy
    yawn_threshold_frames = int(fps * 1)
    head_threshold_frames = int(fps * 0.8)
    
    eye_closed_queue = deque(maxlen=queue_length)
    yawn_queue = deque(maxlen=queue_length)
    head_queue = deque(maxlen=queue_length)
    head_warning_time = None
    yawn_warning_time = None
    drowsy_warning_time = None
    alarm_end_time = None

    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Cannot access webcam.")
        return

    while True:  # Process each frame
        ret, frame = cap.read()  # Read frame
        if not ret:
            print("Failed to retrieve frame.")
            break

        # Preprocess image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
        results = model.predict(source=[img], save=False)[0]  # Prediction result returned as list -> [0]: first result

        # Visualize results and print object info
        detected_event_list = []  # Initialize empty list for detected events
        current_eye_closed = False
        current_yawn = False
        current_head_event = False

        for result in results:  # Process all detected objects
            boxes = result.boxes  # Extract bounding boxes
            xyxy = boxes.xyxy.cpu().numpy()  # Convert bbox coordinates to numpy
            confs = boxes.conf.cpu().numpy()  # Confidence scores
            classes = boxes.cls.cpu().numpy()  # Class IDs

            for i in range(len(xyxy)):  # Loop through detected objects
                xmin, ymin, xmax, ymax = map(int, xyxy[i])
                confidence = confs[i]
                label = int(classes[i])
                
                # Print object info
                print(f"Detected {model.names[label]} with confidence {confidence:.2f} at [{xmin}, {ymin}, {xmax}, {ymax}]")

                if confidence > 0.5:  # Only show objects above threshold
                    label_text = f"{model.names[label]} {confidence:.2f}"

                    # Default color (green)
                    color = (0, 255, 0)

                    # Eye closed state (labels 0,1,2 assumed)
                    if label in [0, 1, 2]:
                        current_eye_closed = True

                    # Head movement (labels 4,5)
                    if label in [4, 5]:
                        color = (0, 255, 255)  # Yellow
                        current_head_event = True

                    # Yawning (label 8)
                    if label == 8:
                        color = (0, 255, 255)  # Yellow
                        current_yawn = True

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Append eye state to queue
        eye_closed_queue.append(current_eye_closed)

        # Append events
        yawn_queue.append(current_yawn)
        head_queue.append(current_head_event)

        # Determine drowsiness
        eye_closed_count = sum(eye_closed_queue)
        if eye_closed_count >= drowsy_threshold_frames:
            detected_event_list.append('drowsy')
            drowsy_warning_time = datetime.datetime.now()
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 3000)
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=3)

        # Yawn detection
        yawn_count = sum(yawn_queue)
        if yawn_count >= yawn_threshold_frames:
            detected_event_list.append('yawn')
            yawn_warning_time = datetime.datetime.now()
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 1000)
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
            yawn_queue.clear()

        # Head movement detection
        head_event_count = sum(head_queue)
        if head_event_count >= head_threshold_frames:
            detected_event_list.append('head_movement')
            head_warning_time = datetime.datetime.now()
            if alarm_end_time is None or datetime.datetime.now() >= alarm_end_time:
                trigger_alarm(True, 'alarm.wav', 1000)
                alarm_end_time = datetime.datetime.now() + datetime.timedelta(seconds=1)
            head_queue.clear()

        if eye_closed_count < drowsy_threshold_frames and yawn_count < yawn_threshold_frames and head_event_count < head_threshold_frames:
            alarm_end_time = None

        current_time = datetime.datetime.now()

        # Change color for eye state
        for result in results:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                xmin, ymin, xmax, ymax = map(int, xyxy[i])
                label = int(classes[i])
                if label in [0, 1, 2]:
                    if 'drowsy' in detected_event_list:
                        color = (0, 0, 255)  # Red
                    else:
                        color = (0, 255, 0)  # Green

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, model.names[label], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display warnings
        font_scale = 0.75
        font_thickness = 2

        if drowsy_warning_time and (current_time - drowsy_warning_time).total_seconds() < WARNING_DURATION:
            cv2.putText(frame, 'Warning: Drowsy Detected!', (50, 150), cv2.FONT_ITALIC, font_scale, (0, 0, 255), font_thickness)
        if yawn_warning_time and (current_time - yawn_warning_time).total_seconds() < WARNING_DURATION:
            cv2.putText(frame, 'Warning: Yawning Detected!', (50, 50), cv2.FONT_ITALIC, font_scale, (0, 255, 255), font_thickness)
        if head_warning_time and (current_time - head_warning_time).total_seconds() < WARNING_DURATION:
            cv2.putText(frame, 'Warning: Head Up/Down Detected!', (50, 100), cv2.FONT_ITALIC, font_scale, (0, 255, 255), font_thickness)

        cv2.imshow('YOLOv8 Webcam Object Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    fps = get_webcam_fps()
    print(f"Webcam FPS: {fps} FPS")
    model_path = 'best.pt'
    model = load_model(model_path)
    webcam_detection(model, fps)