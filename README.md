# TechForge Studio - Real-Time Object Counter

A real-time AI-powered object detection and counting system with a privacy-first face glitch effect, available in two versions: a standalone Python desktop app and a full-stack web app.

## Features
- Live object detection using a pretrained YOLOv8 model
- Real-time bounding boxes with labels for detected objects
- Live count dashboard showing totals per object type
- Privacy-first face glitch effect (pixelation, RGB channel shift, noise bands) automatically applied when a person is detected
- FPS counter for performance tracking (desktop version)

## Versions

### 1. Desktop App (`object_counter.py`)
A standalone Python script using OpenCV for webcam capture and display. Runs entirely on one machine, no browser needed.

**Tech Stack:** Python, OpenCV, Ultralytics YOLOv8, NumPy

**Run it:**
\`\`\`bash
pip install opencv-python ultralytics
python object_counter.py
\`\`\`
Press `q` to quit.

### 2. Full-Stack Web App (`app.py` + `index.html`)
A client-server version with a Flask backend running the AI model and a browser-based frontend for the live camera feed and dashboard.

**Tech Stack:**
- Backend: Python, Flask, Flask-CORS, OpenCV, Ultralytics YOLOv8
- Frontend: HTML, CSS, JavaScript (Canvas API, Fetch API)

**How it works:**
The frontend captures webcam frames and sends them to the backend via a REST API (`/detect` endpoint). The backend runs YOLO detection and face-glitch processing on each frame, then returns the fully processed image along with live object counts as JSON.

**Run it:**
\`\`\`bash
pip install flask flask-cors opencv-python ultralytics
python app.py
\`\`\`
Then open `index.html` in your browser and allow camera access.

## Author
Built by Ayshu (TechForge Studio)
