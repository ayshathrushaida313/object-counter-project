from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import random
from ultralytics import YOLO
from collections import Counter

app = Flask(__name__)
CORS(app)
model = YOLO("yolov8n.pt")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def glitch_face(face_region):
    h, w = face_region.shape[:2]
    if h < 4 or w < 4:
        return face_region

    small = cv2.resize(face_region, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    b, g, r = cv2.split(pixelated)
    shift = max(2, w // 15)
    b = np.roll(b, shift, axis=1)
    r = np.roll(r, -shift, axis=1)
    glitched = cv2.merge([b, g, r])

    for _ in range(6):
        y = random.randint(0, h - 1)
        band_h = random.randint(1, max(1, h // 12))
        y_end = min(h, y + band_h)
        glitched[y:y_end, :] = np.roll(glitched[y:y_end, :], random.randint(-15, 15), axis=1)

    return glitched

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame, verbose=False)[0]
    counts = Counter()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        counts[label] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "person":
            person_roi = frame[max(y1,0):y2, max(x1,0):x2]
            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_roi, 1.1, 5)
            for (fx, fy, fw, fh) in faces:
                face_region = person_roi[fy:fy+fh, fx:fx+fw]
                person_roi[fy:fy+fh, fx:fx+fw] = glitch_face(face_region)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), (255, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    processed_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "image": "data:image/jpeg;base64," + processed_image,
        "counts": dict(counts),
        "total": sum(counts.values())
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)