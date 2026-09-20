import cv2
import time
import numpy as np
import random
from ultralytics import YOLO
from collections import Counter

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

prev_time = 0

def glitch_face(face_region):
    h, w = face_region.shape[:2]
    if h < 4 or w < 4:
        return face_region

    # 1. Pixelate
    small = cv2.resize(face_region, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # 2. RGB channel shift
    b, g, r = cv2.split(pixelated)
    shift = max(2, w // 15)
    b = np.roll(b, shift, axis=1)
    r = np.roll(r, -shift, axis=1)
    glitched = cv2.merge([b, g, r])

    # 3. Random horizontal noise bands
    for _ in range(6):
        y = random.randint(0, h - 1)
        band_h = random.randint(1, max(1, h // 12))
        y_end = min(h, y + band_h)
        glitched[y:y_end, :] = np.roll(glitched[y:y_end, :], random.randint(-15, 15), axis=1)

    return glitched

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    panel_height = 40 + len(counts) * 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (260, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(frame, "TechForge Studio", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    y_offset = 60
    for label, count in counts.items():
        cv2.putText(frame, f"{label}: {count}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y_offset += 25

    total = sum(counts.values())
    cv2.putText(frame, f"Total: {total}", (frame.shape[1] - 160, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow("TechForge Studio - Real-Time Object Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()