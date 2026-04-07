# 🎯 Smart Attendance System using YOLO

## 📌 Overview

This project is an AI-based Smart Attendance System that uses **YOLO (You Only Look Once)** object detection to automatically detect and recognize students in real-time and mark attendance.

It eliminates manual attendance, reduces errors, and improves efficiency in classrooms.

---

## 🚀 Features

* 🎥 Real-time student detection using YOLO
* 🧠 AI-powered face/person recognition
* 📋 Automatic attendance marking
* 💾 Stores attendance records in database/CSV
* ⏱️ Fast and accurate detection
* 📊 Easy report generation

---

## 🛠️ Technologies Used

* Python
* YOLO (You Only Look Once)
* OpenCV
* NumPy
* Pandas

---

## ⚙️ How It Works

1. The system captures live video using a webcam.
2. YOLO model detects persons/faces in the frame.
3. Detected faces are compared with stored data.
4. If a match is found:

   * Student name is identified
   * Attendance is marked automatically
5. Data is stored in a file/database.

---

## 📂 Project Structure

attendance-system-yolo/
│── dataset/
│── code/
│── embeddings/
│── attendance.csv
│── README.md

---

## ▶️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/attendance-system-yolo.git
cd attendance-system-yolo
```

2. Install dependencies:

```bash
pip install opencv-python numpy pandas
```


---

## 📊 Output

* Detects students in real-time
* Displays name on screen
* Automatically updates attendance.csv

---

## 🎯 Future Enhancements

* Add Face Recognition (Deep Learning)
* Cloud database integration
* Mobile app integration
* Mask detection support

---

## 👩‍💻 Author

Nivetha S S

---

## 📜 License

This project is for educational purposes.
