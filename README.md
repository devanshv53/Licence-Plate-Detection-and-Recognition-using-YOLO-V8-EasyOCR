# 🚗 License Plate Detection & Recognition using YOLOv8 + EasyOCR

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange?logo=opencv)
![EasyOCR](https://img.shields.io/badge/EasyOCR-OCR-green)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-red?logo=opencv)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Handling-purple?logo=pandas)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?logo=pytorch)

---

## 📌 Overview
> This project detects vehicles’ license plates in **real time** and recognizes the text using **YOLOv8 (object detection)** and **EasyOCR (optical character recognition).**  
>  
> ✅ Built as a **major project at Thapar Institute of Engineering & Technology (TIET)** with a dataset collected from our campus.  
> ✅ Detected plates are logged into a CSV file with **entry & exit timestamps**, helping track vehicle movement efficiently.  

---

## 🎯 Why this project?
- 🚦 Smart Campus / Smart City Applications → automate vehicle logging at gates/parking.  
- ⚡ Real-time detection with **YOLOv8**.  
- 🔍 Accurate OCR for Indian license plates using **EasyOCR**.  
- 📊 Data logging → exportable to CSV (or SQL databases).  
- 🤖 Blend of **AI + Computer Vision + Data Engineering** showcasing practical ML deployment.  

---

## 🛠️ Tech Stack
- **Python 3.10+**  
- **YOLOv8 (Ultralytics)** – vehicle & license plate detection  
- **EasyOCR** – text extraction from detected plates  
- **OpenCV** – image preprocessing  
- **Pandas** – data handling & CSV export  
- **Hydra** – config management  
- **PyTorch** – deep learning backend  

---

## 📂 Project Structure
```bash
📦 Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR
 ┣ 📜 predictWithOCR.py          # Main script (core pipeline)
 ┣ 📜 vehicle_entry_exit_log.csv # Generated log file
 ┣ 📂 runs/                      # YOLOv8 inference outputs
 ┣ 📂 images/                    # Screenshots & demo outputs
 ┗ 📜 README.md                  # This file

```
-----
## ⚡ How It Works
1. YOLOv8 detects vehicles & license plates in frames/images.  
2. EasyOCR extracts text from cropped plate regions.  
3. Text normalization (uppercase, remove noise).  
4. Entry/Exit events logged into CSV with timestamps.  
5. Logs can be extended to SQL DBs for scalability.

---
## 📸 Screenshots
<img width="1470" height="956" alt="Screenshot 2025-08-26 at 1 45 29 AM" src="https://github.com/user-attachments/assets/ef54eb38-debb-4c07-82aa-fca287169b6d" />
<img width="1470" height="956" alt="Screenshot 2025-08-26 at 1 46 27 AM" src="https://github.com/user-attachments/assets/708a9f1b-480d-4ee3-958b-43f3401cc602" />
<img width="851" height="695" alt="Screenshot 2025-08-26 at 1 47 16 AM" src="https://github.com/user-attachments/assets/277dcce7-e098-4979-b3db-670ae87b2137" />

---
## 📑 Output Example (CSV Log)
| License Plate | Entry Time              | Exit Time              |
|---------------|-------------------------|------------------------|
| PB10AB1234    | 26-08-2025 10:23:45.123 | None                   |
| CH01AA4321    | 26-08-2025 10:25:12.789 | 26-08-2025 11:02:33.456 |

---
## ▶️ Running the Project
1. Install dependencies :

   pip install ultralytics easyocr opencv-python pandas hydra-core torch

3. Run detection
python predictWithOCR.py source=path/to/video.mp4

4. Check logs
- vehicle_entry_exit_log.csv (log file)
- runs/ (YOLO annotated outputs)

----
## 🚀 Future Enhancements
- ✅ Store logs in SQL DB (SQLite/MySQL).  
- ✅ Deploy as a Flask/FastAPI web app.  
- ✅ Integrate with IoT devices (boom barriers).  
- ✅ Add GUI dashboard for vehicle activity.  
