import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_distances
import os

# -------------------- CONFIG --------------------
THRESHOLD = 0.45   # lower = stricter (0.35–0.5 ideal)
CAMERA_URL = "http://10.144.161.7:8080/video"
# ------------------------------------------------

# Load YOLO (used only to locate faces roughly)
model = YOLO("yolov8n.pt")

# Load embeddings
with open("embeddings/face_embeddings.pkl", "rb") as f:
    known_embeddings = pickle.load(f)

# Camera
cap = cv2.VideoCapture(CAMERA_URL)

# Attendance file
attendance_file = "attendance/attendance.csv"
os.makedirs("attendance", exist_ok=True)

if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(attendance_file, index=False)

marked = set()

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            try:
                embedding = DeepFace.represent(
                    img_path=face,
                    model_name="Facenet",
                    enforce_detection=False
                )[0]["embedding"]
            except:
                continue

            # --------- MATCHING LOGIC (FIXED) ----------
            min_distance = 1.0
            identity = "Unknown"

            for person, emb_list in known_embeddings.items():
                for saved_emb in emb_list:
                    dist = cosine_distances(
                        [embedding],
                        [saved_emb]
                    )[0][0]

                    if dist < min_distance:
                        min_distance = dist
                        identity = person

            if min_distance > THRESHOLD:
                identity = "Unknown"
            # --------------------------------------------

            # Attendance marking
            if identity != "Unknown" and identity not in marked:
                time = datetime.now().strftime("%H:%M:%S")
                df = pd.read_csv(attendance_file)
                df.loc[len(df)] = [identity, time]
                df.to_csv(attendance_file, index=False)
                marked.add(identity)

            # Draw box & label
            color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{identity}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
