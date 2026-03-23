"""
Simple inference script to test the trained model on images or videos
"""

from ultralytics import YOLO
import cv2
import os
import argparse
from pathlib import Path

def test_on_image(model_path, image_path, output_dir='outputs', confidence=0.5):
    """
    Test model on a single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to test image
        output_dir: Directory to save results
        confidence: Confidence threshold
    """
    
    # Load model
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    results = model(image_path, conf=confidence)
    
    # Get annotated image
    annotated = results[0].plot()
    
    # Save result
    output_path = os.path.join(output_dir, f"result_{Path(image_path).name}")
    cv2.imwrite(output_path, annotated)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print detections
    boxes = results[0].boxes
    if len(boxes) > 0:
        print(f"\nDetections:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  - Class {cls}: {conf:.2f}")
    else:
        print("\nNo detections found")
    
    return annotated


def test_on_video(model_path, video_path, output_dir='outputs', confidence=0.5):
    """
    Test model on video file
    
    Args:
        model_path: Path to trained model
        video_path: Path to test video
        output_dir: Directory to save results
        confidence: Confidence threshold
    """
    
    # Load model
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    output_path = os.path.join(output_dir, f"result_{Path(video_path).name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=confidence, verbose=False)
        
        # Get annotated frame
        annotated = results[0].plot()
        
        # Write frame
        out.write(annotated)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames...")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"  Total frames processed: {frame_count}")


def test_on_webcam(model_path, confidence=0.5):
    """
    Test model on webcam feed
    
    Args:
        model_path: Path to trained model
        confidence: Confidence threshold
    """
    
    # Load model
    model = YOLO(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Testing on webcam... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=confidence, verbose=False)
        
        # Get annotated frame
        annotated = results[0].plot()
        
        # Display
        cv2.imshow('Driver Distraction Detection - Press Q to quit', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Test YOLOv8 Driver Distraction Detection Model')
    parser.add_argument('--model', type=str, default='best.pt', 
                       help='Path to trained model')
    parser.add_argument('--source', type=str, default='webcam',
                       help='Source: image path, video path, or "webcam"')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Please train the model first using train_distraction_model.py")
        return
    
    print("="*60)
    print("YOLOv8 Driver Distraction Detection - Testing")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Confidence: {args.conf}")
    print(f"Source: {args.source}")
    print("="*60)
    
    # Determine source type and run appropriate test
    if args.source.lower() == 'webcam':
        test_on_webcam(args.model, args.conf)
    
    elif os.path.isfile(args.source):
        # Check if image or video
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        file_ext = Path(args.source).suffix.lower()
        
        if file_ext in image_extensions:
            test_on_image(args.model, args.source, args.output, args.conf)
        elif file_ext in video_extensions:
            test_on_video(args.model, args.source, args.output, args.conf)
        else:
            print(f"Error: Unsupported file format: {file_ext}")
    
    else:
        print(f"Error: Source '{args.source}' not found!")


if __name__ == "__main__":
    # If run without arguments, show examples
    import sys
    if len(sys.argv) == 1:
        print("="*60)
        print("YOLOv8 Driver Distraction Detection - Test Script")
        print("="*60)
        print("\nUsage Examples:")
        print("\n1. Test on webcam:")
        print("   python test_model.py --model best.pt --source webcam")
        print("\n2. Test on image:")
        print("   python test_model.py --model best.pt --source image.jpg")
        print("\n3. Test on video:")
        print("   python test_model.py --model best.pt --source video.mp4")
        print("\n4. Adjust confidence:")
        print("   python test_model.py --model best.pt --source webcam --conf 0.3")
        print("\n" + "="*60)
    else:
        main()
