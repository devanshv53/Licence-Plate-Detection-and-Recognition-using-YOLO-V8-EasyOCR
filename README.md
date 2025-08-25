# Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR
_1-Good
🚗 License Plate Detection & Recognition using YOLOv8 + EasyOCR

📌 Overview

This project detects vehicles’ license plates in real time and recognizes the text using YOLOv8 (for object detection) and EasyOCR (for optical character recognition).

It was built as a major project at Thapar Institute of Engineering & Technology (TIET) using a real-time dataset collected from our campus.

Detected license plates are logged into a CSV file with entry and exit timestamps, helping track vehicle movement efficiently.

🎯 Why this project?

🚦 Smart Campus/Smart City Applications → automate vehicle logging at gates/parking.

⏱️ Real-time detection with YOLOv8.

🔤 Accurate OCR for Indian license plates using EasyOCR.

📊 Data logging → exportable to CSV (or can be extended to SQL databases).

💡 A blend of AI + Computer Vision + Data Engineering, showcasing practical ML deployment.

🛠️ Tech Stack

Python 3.10+

YOLOv8 (Ultralytics) – vehicle & license plate detection

EasyOCR – text extraction from detected plates

OpenCV – image preprocessing

Pandas – data handling & CSV export

Hydra – config management

PyTorch – deep learning backend

📂 Project Structure
📦 Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR
 ┣ 📜 predictWithOCR.py    # Main script (core pipeline)
 ┣ 📜 vehicle_entry_exit_log.csv   # Generated log file
 ┣ 📂 runs/                # YOLOv8 inference outputs
 ┣ 📂 images/              # Screenshots & demo outputs
 ┗ 📜 README.md            # This file

⚡ How It Works

YOLOv8 detects vehicles & license plates in video frames/images.

EasyOCR extracts text from cropped plate regions.

Text is normalized (removing noise, uppercase formatting).

Entry/Exit events logged into a CSV file with timestamps.

Results can be extended to SQL databases for scalable storage.

📸 Screenshots

Detection in action:






📑 Output Example (CSV Log)
License Plate,Entry Time,Exit Time
PB10AB1234,26-08-2025 10:23:45.123,None
CH01AA4321,26-08-2025 10:25:12.789,26-08-2025 11:02:33.456

▶️ Running the Project
1️⃣ Install dependencies
pip install ultralytics easyocr opencv-python pandas hydra-core torch

2️⃣ Run detection
python predictWithOCR.py source=path/to/video.mp4

3️⃣ Check logs

CSV file generated: vehicle_entry_exit_log.csv

YOLO annotated video/images saved in runs/

🚀 Future Enhancements

✅ Store logs in SQL database (SQLite/MySQL) instead of CSV.

✅ Deploy as a Flask/FastAPI web app for gate monitoring.

✅ Integrate with IoT devices (boom barriers, smart gates).

✅ Add GUI dashboard to visualize vehicle activity.

📜 Dataset

Real-time dataset collected from TIET (Thapar Institute of Engineering & Technology) campus gates.

Includes various lighting conditions, angles, and Indian number plate formats.

