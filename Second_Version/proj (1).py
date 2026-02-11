import cv2
import numpy as np
import time
import pyttsx3
import threading
import pygame
import tempfile
from scipy.io import wavfile
from scipy.spatial import distance as dist

# Configurable thresholds
EYE_AR_THRESH = 0.25  # Eye aspect ratio threshold
EYE_AR_CONSEC_FRAMES = 15  # Consecutive frames for drowsiness
DROWSY_THRESH = 0.28  # Threshold for drowsiness warning
DISTRACTION_WARNING_TIME = 5  # seconds
DISTRACTION_ALARM_TIME = 10  # seconds
ALERT_COOLDOWN = 3  # seconds

class AlarmSystem:
    """Handles audio alerts with threading"""
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        self.is_playing = False
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self.alarm_sound_path = self._create_alarm_sound()
            if self.alarm_sound_path:
                self.alarm_sound = pygame.mixer.Sound(self.alarm_sound_path)
                self.alarm_sound.set_volume(1.0)
            else:
                self.alarm_sound = None
        except Exception as e:
            print(f"Warning: Audio initialization failed: {e}")
            self.alarm_sound = None
    
    def _create_alarm_sound(self):
        """Generate alarm beep sound"""
        try:
            sample_rate = 44100
            duration = 0.5
            frequency = 1000
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            beep = np.sin(2 * np.pi * frequency * t)
            silence = np.zeros(int(sample_rate * 0.1))
            alarm = np.concatenate([beep, silence, beep, silence, beep])
            alarm = np.int16(alarm * 32767)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            wavfile.write(temp_file.name, sample_rate, alarm)
            return temp_file.name
        except Exception as e:
            print(f"Warning: Could not create alarm sound: {e}")
            return None
    
    def play_voice_alert(self, message):
        """Play TTS alert"""
        if not self.is_playing:
            self.is_playing = True
            thread = threading.Thread(target=self._speak, args=(message,))
            thread.daemon = True
            thread.start()
    
    def _speak(self, message):
        """TTS handler"""
        try:
            self.engine.say(message)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.is_playing = False
    
    def play_alarm(self):
        """Play alarm sound"""
        if self.alarm_sound:
            thread = threading.Thread(target=self._play_sound)
            thread.daemon = True
            thread.start()
    
    def _play_sound(self):
        """Sound playback handler"""
        try:
            self.alarm_sound.play()
            while pygame.mixer.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Alarm error: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            pygame.mixer.quit()
        except:
            pass

def eye_aspect_ratio(eye):
    """Calculate eye aspect ratio"""
    # Vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horizontal eye landmark
    C = dist.euclidean(eye[0], eye[3])
    
    ear = (A + B) / (2.0 * C)
    return ear

def draw_info_panel(frame, w, h, stats):
    """Draw statistics panel"""
    panel_height = 200
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    y_offset = 30
    line_height = 30
    
    info_lines = [
        f"Session: {int(stats['session_time'])}s",
        f"Distractions: {stats['distraction_count']}",
        f"Warnings: {stats['haptic_count']}",
        f"Alarms: {stats['alarm_count']}",
        f"EAR: {stats['ear']:.2f}",
    ]
    
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (10, y_offset + i * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def detect_eyes_simple(gray, face_roi):
    """Simple eye detection using template matching"""
    # Create eye cascade detector
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    eyes = eye_cascade.detectMultiScale(
        face_roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return eyes

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Initialize alarm system
    alarm_system = AlarmSystem()
    
    # Counters
    distraction_timer = 0
    distraction_count = 0
    haptic_count = 0
    alarm_count = 0
    drowsy_frames = 0
    closed_frames = 0
    last_update = time.time()
    last_alert_time = 0
    start_time = time.time()
    
    haptic_triggered = False
    alarm_triggered = False
    
    # For smoothing
    ear_history = []
    max_history = 5
    
    print("=" * 60)
    print("DRIVER MONITORING SYSTEM - Lightweight Version")
    print("=" * 60)
    print("Uses: OpenCV Haar Cascades (No external model files needed)")
    print("\nControls:")
    print("  Q - Quit and show summary")
    print("  R - Reset counters")
    print("\nMonitoring started...")
    print("=" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read from camera")
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(120, 120)
            )
            
            current_time = time.time()
            distracted = True
            eyes_closed = False
            drowsy = False
            avg_ear = 0
            
            if len(faces) > 0:
                # Use the largest face
                face = max(faces, key=lambda rect: rect[2] * rect[3])
                (x, y, fw, fh) = face
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 0, 0), 2)
                
                # Extract face ROI
                face_roi_gray = gray[y:y+fh, x:x+fw]
                face_roi_color = frame[y:y+fh, x:x+fw]
                
                # Detect eyes
                eyes = eye_cascade.detectMultiScale(
                    face_roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=10,
                    minSize=(20, 20)
                )
                
                if len(eyes) >= 2:
                    # Sort eyes by x-coordinate (left to right)
                    eyes = sorted(eyes, key=lambda e: e[0])
                    
                    # Take first two eyes (left and right)
                    left_eye = eyes[0]
                    right_eye = eyes[1] if len(eyes) > 1 else eyes[0]
                    
                    # Draw eye rectangles
                    for (ex, ey, ew, eh) in [left_eye, right_eye]:
                        cv2.rectangle(face_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    
                    # Calculate simple EAR approximation using eye rectangles
                    # Height/Width ratio as proxy for eye openness
                    left_ear = left_eye[3] / (left_eye[2] + 1e-6)
                    right_ear = right_eye[3] / (right_eye[2] + 1e-6)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Smooth EAR
                    ear_history.append(avg_ear)
                    if len(ear_history) > max_history:
                        ear_history.pop(0)
                    avg_ear = np.mean(ear_history)
                    
                    # Check eye status
                    if avg_ear < EYE_AR_THRESH:
                        closed_frames += 1
                        if closed_frames >= EYE_AR_CONSEC_FRAMES:
                            eyes_closed = True
                    else:
                        closed_frames = 0
                    
                    # Check drowsiness
                    if avg_ear < DROWSY_THRESH and avg_ear >= EYE_AR_THRESH:
                        drowsy_frames += 1
                        if drowsy_frames >= EYE_AR_CONSEC_FRAMES:
                            drowsy = True
                    else:
                        drowsy_frames = 0
                    
                    # Check if eyes are detected (not distracted)
                    if len(eyes) >= 2 and not eyes_closed:
                        # Check if face is centered
                        face_center_x = x + fw // 2
                        frame_center_x = w // 2
                        
                        # Allow 30% deviation from center
                        if abs(face_center_x - frame_center_x) < w * 0.3:
                            distracted = False
            
            # Handle drowsiness alerts
            if drowsy and (current_time - last_alert_time) > ALERT_COOLDOWN:
                cv2.putText(frame, "!!! DROWSINESS DETECTED !!!", 
                           (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(frame, "WAKE UP!", 
                           (w//2 - 80, h - 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                alarm_system.play_voice_alert("Alert! Driver drowsiness detected!")
                last_alert_time = current_time
            
            # Update counters every 0.5s
            if current_time - last_update >= 0.5:
                if distracted or eyes_closed or drowsy:
                    distraction_timer += 1
                    distraction_count += 1
                    
                    if distraction_timer > DISTRACTION_WARNING_TIME and not haptic_triggered:
                        haptic_count += 1
                        haptic_triggered = True
                        alarm_system.play_voice_alert("Warning! Pay attention!")
                    
                    if distraction_timer > DISTRACTION_ALARM_TIME and not alarm_triggered:
                        alarm_count += 1
                        alarm_triggered = True
                        alarm_system.play_alarm()
                        alarm_system.play_voice_alert("Danger! Focus on road!")
                else:
                    distraction_timer = 0
                    haptic_triggered = False
                    alarm_triggered = False
                
                last_update = current_time
            
            # Determine status
            if drowsy:
                status = "DROWSY"
                color = (0, 165, 255)
            elif eyes_closed:
                status = "EYES CLOSED"
                color = (0, 0, 255)
            elif distracted:
                status = "DISTRACTED"
                color = (0, 0, 255)
            else:
                status = "FOCUSED"
                color = (0, 255, 0)
            
            # Draw status
            cv2.putText(frame, f"Status: {status}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Draw warnings
            if distraction_timer > 0:
                cv2.putText(frame, f"DISTRACTED: {distraction_timer}s", 
                           (20, h - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if distraction_timer > DISTRACTION_WARNING_TIME:
                cv2.putText(frame, ">>> WARNING <<<", (w - 280, h - 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            if distraction_timer > DISTRACTION_ALARM_TIME:
                cv2.putText(frame, "!!! ALARM !!!", (w - 220, h - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                
                # Flash effect
                if int(current_time * 2) % 2 == 0:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), 30)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw info panel
            stats = {
                'session_time': current_time - start_time,
                'distraction_count': distraction_count,
                'haptic_count': haptic_count,
                'alarm_count': alarm_count,
                'ear': avg_ear
            }
            draw_info_panel(frame, w, h, stats)
            
            # Show frame
            cv2.imshow("Driver Monitor - Lightweight", frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                distraction_count = 0
                haptic_count = 0
                alarm_count = 0
                start_time = time.time()
                print("\nCounters reset!")
    
    finally:
        # Show summary
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Total Time:       {int(total_time)}s ({total_time/60:.1f} min)")
        print(f"Distractions:     {distraction_count}")
        print(f"Warnings:         {haptic_count}")
        print(f"Alarms:           {alarm_count}")
        
        if total_time > 0:
            print(f"Distraction Rate: {(distraction_count/total_time)*60:.1f}/min")
        
        print("=" * 60)
        
        cap.release()
        cv2.destroyAllWindows()
        alarm_system.cleanup()
        
        # Cleanup temp file
        if hasattr(alarm_system, 'alarm_sound_path') and alarm_system.alarm_sound_path:
            try:
                import os
                os.unlink(alarm_system.alarm_sound_path)
            except:
                pass

if __name__ == "__main__":
    main()