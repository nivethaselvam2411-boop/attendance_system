import cv2
import os
from ultralytics import YOLO

# -----------------------------
# SETTINGS - CHANGE THESE
# -----------------------------
# Your mobile camera IP stream URL
MOBILE_CAM_URL = "http://10.156.115.152:8080/video"  # Replace with your phone IP
DATASET_PATH = "dataset"
NUM_IMAGES = 10
YOLO_MODEL_PATH = "yolov8n.pt"  # Make sure you have this model

# -----------------------------
# INPUT PERSON NAME
# -----------------------------
name = input("Enter person's name: ").strip()
person_path = os.path.join(DATASET_PATH, name)
os.makedirs(person_path, exist_ok=True)

# -----------------------------
# LOAD YOLO FACE MODEL
# -----------------------------
print("Loading YOLO face detection model...")
model = YOLO(YOLO_MODEL_PATH)

# -----------------------------
# START CAPTURING
# -----------------------------
cap = cv2.VideoCapture(MOBILE_CAM_URL)
count = 0
print(f"Starting capture for {name}. Press 'q' to quit anytime.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame. Check your mobile camera URL!")
        break

    # Detect faces using YOLO
    results = model(frame)

    # Draw bounding boxes and crop face
    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Captured {count}/{NUM_IMAGES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Mobile Camera - Press 'c' to capture", frame)
    key = cv2.waitKey(1) & 0xFF

    # Capture image
    if key == ord('c'):
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Save cropped face
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                face_img = frame[y1:y2, x1:x2]
                img_name = os.path.join(person_path, f"{name}_{count}.jpg")
                cv2.imwrite(img_name, face_img)
                print(f"Captured image {count+1}/{NUM_IMAGES}")
                count += 1
                break  # Save only first detected face

        else:
            print("No face detected, try again.")

        if count >= NUM_IMAGES:
            print(f"Captured all {NUM_IMAGES} images for {name}!")
            break

    elif key == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
