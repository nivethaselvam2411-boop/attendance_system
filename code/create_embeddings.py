import cv2
import pickle
import pandas as pd
from datetime import datetime
from mtcnn import MTCNN
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_distances
import os

# -------- SETTINGS --------
THRESHOLD = 0.45
CAMERA_URL = "http://10.144.161.7:8080/video"  # mobile cam or 0 for laptop
# --------------------------

detector = MTCNN()

with open("embeddings/face_embeddings.pkl", "rb") as f:
    known_embeddings = pickle.load(f)

os.makedirs("attendance", exist_ok=True)
attendance_file = "attendance/attendance.csv"

if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(attendance_file, index=False)

cap = cv2.VideoCapture(CAMERA_URL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    faces = detector.detect_faces(frame)

    for face_data in faces:
        x, y, w, h = face_data['box']
        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        try:
            emb = DeepFace.represent(
                face,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]
        except:
            continue

        min_dist = 1.0
        identity = "Unknown"

        # 🔁 Compare with ALL persons
        for person, emb_list in known_embeddings.items():
            for known_emb in emb_list:
                dist = cosine_distances([emb], [known_emb])[0][0]
                if dist < min_dist:
                    min_dist = dist
                    identity = person

        if min_dist > THRESHOLD:
            identity = "Unknown"

        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%H:%M:%S")

        # ✅ Mark attendance for MULTIPLE persons
        if identity != "Unknown":
            df = pd.read_csv(attendance_file)
            already_marked = (
                (df["Name"] == identity) &
                (df["Date"] == today)
            ).any()

            if not already_marked:
                df.loc[len(df)] = [identity, today, now]
                df.to_csv(attendance_file, index=False)

        color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, identity, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Multi-Person Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
