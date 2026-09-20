\# TechForge Studio - Real-Time Object Counter



A real-time AI-powered object detection and counting system built with Python, OpenCV, and YOLOv8.



\## Features

\- Live webcam object detection using a pretrained YOLOv8 model

\- Real-time bounding boxes with labels for detected objects

\- Live count dashboard showing totals per object type

\- FPS counter for performance tracking

\- Privacy-first face glitch effect applied automatically when a person is detected



\## Tech Stack

\- Python

\- OpenCV

\- Ultralytics YOLOv8

\- NumPy



\## How It Works

The system captures live video from a webcam, runs each frame through a pretrained YOLOv8 model to detect objects, and overlays bounding boxes, labels, and a live count dashboard. When a person is detected, their face is automatically located and replaced with a glitch effect (pixelation, RGB channel shift, and noise bands) for privacy.



\## Setup

\\`\\`\\`bash

pip install opencv-python ultralytics

python object\_counter.py

\\`\\`\\`



Press `q` to quit.



\## Author

Built by Ayshu (TechForge Studio)

