import cv2
import numpy as np
from ultralytics import YOLO
import time

class LicensePlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.prev_detections = {}
        self.prev_time = time.time()

    def detect_and_estimate(self, frame):
        current_time = time.time()
        results = self.model(frame)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center of the bounding box
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Estimate speed
                plate_id = f"{x1}_{y1}_{x2}_{y2}"
                if plate_id in self.prev_detections:
                    prev_center, prev_time = self.prev_detections[plate_id]
                    distance = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                    time_diff = current_time - prev_time
                    speed = distance / time_diff  # pixels per second
                    
                    # Convert to km/h (this conversion factor needs to be calibrated)
                    speed_kmh = speed * 0.1  # Example conversion factor
                    
                    cv2.putText(frame, f"Speed: {speed_kmh:.2f} km/h", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                
                self.prev_detections[plate_id] = (center, current_time)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        self.prev_time = current_time
        return frame

def main():
    detector = LicensePlateDetector('"C:\Users\sambh\Downloads\best.pt"')
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detector.detect_and_estimate(frame)
        cv2.imshow('License Plate Detection and Speed Estimation', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()