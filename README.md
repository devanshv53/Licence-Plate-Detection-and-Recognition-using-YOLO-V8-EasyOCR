<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>License Plate Project README</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f7f9fc;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 900px;
            margin: auto;
            padding: 2rem;
        }
        .section {
            background-color: #fff;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 2rem;
        }
        .header {
            text-align: center;
            padding-bottom: 1rem;
            border-bottom: 2px solid #e2e8f0;
            margin-bottom: 2rem;
        }
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1a202c;
            line-height: 1.2;
            margin-bottom: 0.5rem;
        }
        .header p {
            font-size: 1.2rem;
            color: #6a737d;
        }
        h2 {
            font-size: 1.75rem;
            font-weight: 600;
            color: #2d3748;
            border-left: 4px solid #4299e1;
            padding-left: 1rem;
            margin-bottom: 1.5rem;
        }
        h3 {
            font-size: 1.25rem;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 0.75rem;
        }
        .live-demo a {
            color: #fff;
            background-color: #4299e1;
            padding: 0.75rem 2rem;
            border-radius: 9999px;
            text-decoration: none;
            font-weight: 600;
            transition: background-color 0.3s ease;
            display: inline-block;
            box-shadow: 0 4px 6px rgba(66, 153, 225, 0.2);
        }
        .live-demo a:hover {
            background-color: #2b6cb0;
        }
        .screenshots {
            display: grid;
            gap: 1.5rem;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        }
        .screenshots img {
            width: 100%;
            height: auto;
            border-radius: 0.75rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .screenshots img:hover {
            transform: scale(1.02);
        }
        .tech-stack li {
            background-color: #edf2f7;
            border-left: 3px solid #63b3ed;
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        pre {
            background-color: #1a202c;
            color: #e2e8f0;
            padding: 1.5rem;
            border-radius: 0.75rem;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        code {
            font-family: 'Courier New', Courier, monospace;
        }
        .terminal-prompt {
            color: #90cdf4;
            display: block;
            margin-bottom: 0.5rem;
        }
        .terminal-output {
            color: #c6f6d5;
        }
        .table-container {
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: #fff;
            border-radius: 0.75rem;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
            margin-top: 1rem;
        }
        th, td {
            text-align: left;
            padding: 1rem;
            border-bottom: 1px solid #e2e8f0;
        }
        th {
            background-color: #e2e8f0;
            font-weight: 600;
            color: #4a5568;
            text-transform: uppercase;
        }
        tr:last-child td {
            border-bottom: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>_1-Good 🚗 License Plate Detection & Recognition using YOLOv8 + EasyOCR</h1>
            <p><strong>📌 Overview</strong></p>
            <p>This project detects vehicles’ license plates in real time and recognizes the text using YOLOv8 (for object detection) and EasyOCR (for optical character recognition).</p>
            <p>It was built as a major project at Thapar Institute of Engineering & Technology (TIET) using a real-time dataset collected from our campus.</p>
            <p>Detected license plates are logged into a CSV file with entry and exit timestamps, helping track vehicle movement efficiently.</p>
        </div>

        <div class="section">
            <h2>🎯 Why this project?</h2>
            <ul class="list-disc pl-5 text-gray-600 space-y-2">
                <li>🚦 Smart Campus/Smart City Applications → automate vehicle logging at gates/parking.</li>
                <li>⏱️ Real-time detection with YOLOv8.</li>
                <li>🔤 Accurate OCR for Indian license plates using EasyOCR.</li>
                <li>📊 Data logging → exportable to CSV (or can be extended to SQL databases).</li>
                <li>💡 A blend of AI + Computer Vision + Data Engineering, showcasing practical ML deployment.</li>
            </ul>
        </div>

        <div class="section">
            <h2>🛠️ Tech Stack</h2>
            <ul class="tech-stack list-none p-0">
                <li><strong>Python 3.10+</strong></li>
                <li><strong>YOLOv8 (Ultralytics)</strong> – vehicle & license plate detection</li>
                <li><strong>EasyOCR</strong> – text extraction from detected plates</li>
                <li><strong>OpenCV</strong> – image preprocessing</li>
                <li><strong>Pandas</strong> – data handling & CSV export</li>
                <li><strong>Hydra</strong> – config management</li>
                <li><strong>PyTorch</strong> – deep learning backend</li>
            </ul>
        </div>

        <div class="section">
            <h2>📂 Project Structure</h2>
            <pre><code>📦 Licence-Plate-Detection-and-Recognition-using-YOLO-V8-EasyOCR
 ┣ 📜 predictWithOCR.py    # Main script (core pipeline)
 ┣ 📜 vehicle_entry_exit_log.csv    # Generated log file
 ┣ 📂 runs/                   # YOLOv8 inference outputs
 ┣ 📂 images/                 # Screenshots & demo outputs
 ┗ 📜 README.md               # This file
</code></pre>
        </div>

        <div class="section">
            <h2>⚡ How It Works</h2>
            <ul class="list-disc pl-5 text-gray-600 space-y-2">
                <li>YOLOv8 detects vehicles & license plates in video frames/images.</li>
                <li>EasyOCR extracts text from cropped plate regions.</li>
                <li>Text is normalized (removing noise, uppercase formatting).</li>
                <li>Entry/Exit events logged into a CSV file with timestamps.</li>
                <li>Results can be extended to SQL databases for scalable storage.</li>
            </ul>
        </div>

        <div class="section">
            <h2>📸 Screenshots</h2>
            <p>Detection in action:</p>
            <div class="screenshots">
                <img src="https://placehold.co/500x300/e2e8f0/1a202c?text=Detection%20in%20Action%201" alt="Detection Screenshot 1">
                <img src="https://placehold.co/500x300/e2e8f0/1a202c?text=Detection%20in%20Action%202" alt="Detection Screenshot 2">
                <img src="https://placehold.co/500x300/e2e8f0/1a202c?text=Detection%20in%20Action%203" alt="Detection Screenshot 3">
            </div>
        </div>

        <div class="section">
            <h2>📑 Output Example (CSV Log)</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>License Plate</th>
                            <th>Entry Time</th>
                            <th>Exit Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>PB10AB1234</td>
                            <td>26-08-2025 10:23:45.123</td>
                            <td>None</td>
                        </tr>
                        <tr>
                            <td>CH01AA4321</td>
                            <td>26-08-2025 10:25:12.789</td>
                            <td>26-08-2025 11:02:33.456</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <h2>▶️ Running the Project</h2>
            <p><strong>1️⃣ Install dependencies</strong></p>
            <pre><code class="language-bash">pip install ultralytics easyocr opencv-python pandas hydra-core torch
</code></pre>
            <p class="mt-4"><strong>2️⃣ Run detection</strong></p>
            <pre><code class="language-bash">python predictWithOCR.py source=path/to/video.mp4
</code></pre>
            <p class="mt-4"><strong>3️⃣ Check logs</strong></p>
            <ul class="list-disc pl-5 text-gray-600 space-y-2">
                <li>CSV file generated: <code>vehicle_entry_exit_log.csv</code></li>
                <li>YOLO annotated video/images saved in <code>runs/</code></li>
            </ul>
        </div>

        <div class="section">
            <h2>🚀 Future Enhancements</h2>
            <ul class="list-disc pl-5 text-gray-600 space-y-2">
                <li>✅ Store logs in SQL database (SQLite/MySQL) instead of CSV.</li>
                <li>✅ Deploy as a Flask/FastAPI web app for gate monitoring.</li>
                <li>✅ Integrate with IoT devices (boom barriers, smart gates).</li>
                <li>✅ Add GUI dashboard to visualize vehicle activity.</li>
            </ul>
        </div>

        <div class="section">
            <h2>📜 Dataset</h2>
            <p>Real-time dataset collected from TIET (Thapar Institute of Engineering & Technology) campus gates.</p>
            <p>Includes various lighting conditions, angles, and Indian number plate formats.</p>
        </div>

    </div>
</body>
</html>
