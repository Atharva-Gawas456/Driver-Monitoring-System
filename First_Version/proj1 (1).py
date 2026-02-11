import cv2
import numpy as np
import time

# Thresholds
EYE_AR_THRESH = 0.25
DROWSY_FRAMES = 15
DISTRACTION_WARN = 5
DISTRACTION_ALARM = 10

def eye_aspect_ratio(eye_h, eye_w):
    """Calculate eye openness ratio"""
    return eye_h / (eye_w + 1e-6)

def main():
    # Initialize
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Counters
    distraction_timer = 0
    drowsy_count = 0
    last_update = time.time()
    
    print("Driver Monitor Started. Press 'Q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))
        
        current_time = time.time()
        distracted = True
        status = "NO FACE"
        color = (0, 0, 255)
        
        if len(faces) > 0:
            # Get largest face
            (x, y, fw, fh) = max(faces, key=lambda r: r[2] * r[3])
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 0, 0), 2)
            
            # Detect eyes in face region
            face_roi = gray[y:y+fh, x:x+fw]
            eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 10, minSize=(20, 20))
            
            if len(eyes) >= 2:
                # Calculate eye aspect ratio
                eyes = sorted(eyes, key=lambda e: e[0])[:2]
                ear = np.mean([eye_aspect_ratio(e[3], e[2]) for e in eyes])
                
                # Draw eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame[y:y+fh, x:x+fw], (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                
                # Check status
                if ear < EYE_AR_THRESH:
                    drowsy_count += 1
                    if drowsy_count >= DROWSY_FRAMES:
                        status = "DROWSY"
                        color = (0, 165, 255)
                        print('\a')  # Beep
                    else:
                        status = "CLOSING"
                        color = (0, 200, 200)
                else:
                    drowsy_count = 0
                    status = "ALERT"
                    color = (0, 255, 0)
                    distracted = False
        
        # Update distraction timer
        if current_time - last_update >= 1:
            if distracted:
                distraction_timer += 1
            else:
                distraction_timer = 0
            last_update = current_time
        
        # Display status
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Show warnings
        if distraction_timer >= DISTRACTION_WARN:
            cv2.putText(frame, f"WARNING: {distraction_timer}s", (20, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        if distraction_timer >= DISTRACTION_ALARM:
            cv2.putText(frame, "!!! ALARM !!!", (w - 250, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            print('\a')  # Beep
            
            # Flash border
            if int(current_time * 2) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 20)
        
        cv2.imshow("Driver Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()