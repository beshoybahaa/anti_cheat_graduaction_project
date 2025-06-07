import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import os
import time
import csv
from collections import deque
from datetime import datetime

# --- CONFIGURATION ---

YOLO_MODEL = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.6
ALARM_COOLDOWN = 10  # seconds per indicator
FRAME_SKIP = 2       # process every Nth frame for efficiency
PRESENCE_HISTORY_SIZE = 7

# Classes: COCO indices for person and cheating-related objects
class_names = {
    0: "person",
    67: "cell phone",
    73: "toaster",  # Toaster instead of book
}
cheating_classes = [67, 73]

ALARM_SOUND = True
last_alarm_time = 0

def trigger_alarm():
    global last_alarm_time
    if ALARM_SOUND and (time.time() - last_alarm_time > ALARM_COOLDOWN):
        try:
            os.system('say "Cheating detected!"')  # macOS
            # For Windows: import winsound; winsound.Beep(1000, 500)
            # For Linux: os.system('spd-say "Cheating detected!"')
        except Exception as e:
            print("Alarm error:", e)
        last_alarm_time = time.time()

# --- INITIALIZE MODELS ---

yolo = YOLO(YOLO_MODEL)
mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- LOGGING FUNCTION WITH METADATA ---

def log_event(indicator, frame, detection_metadata=None):
    ts = int(time.time())
    iso_timestamp = datetime.utcnow().isoformat()
    filename = f"cheating_{ts}.jpg"
    cv2.imwrite(filename, frame)
    with open("cheating_log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if f.tell() == 0:
            writer.writerow([
                "unix_timestamp", "iso_timestamp", "indicator", 
                "filename", "detected_classes", "confidences", "boxes"
            ])
        detected_classes = []
        confidences = []
        boxes = []
        if detection_metadata:
            detected_classes = detection_metadata.get("detected_classes", [])
            confidences = detection_metadata.get("confidences", [])
            boxes = detection_metadata.get("boxes", [])
        writer.writerow([
            ts, iso_timestamp, indicator, filename, 
            "|".join(map(str, detected_classes)),
            "|".join("{:.3f}".format(c) for c in confidences),
            "|".join(str(b) for b in boxes)
        ])

# --- COOLDOWN MANAGEMENT ---

indicator_cooldowns = {}

def should_trigger(indicator):
    now = time.time()
    last = indicator_cooldowns.get(indicator, 0)
    if now - last > ALARM_COOLDOWN:
        indicator_cooldowns[indicator] = now
        return True
    return False

# --- TEMPORAL WINDOW MANAGEMENT (3 seconds, 35%) ---

FPS_ESTIMATE = 10  # Set to your processed FPS. Adjust as needed!
WINDOW_SECONDS = 3
WINDOW_FRAMES = WINDOW_SECONDS * FPS_ESTIMATE
THRESHOLD_RATIO = 0.35  # 35%

indicator_windows = {
    "No person detected": deque([False]*WINDOW_FRAMES, maxlen=WINDOW_FRAMES),
    "Multiple persons detected (YOLO)": deque([False]*WINDOW_FRAMES, maxlen=WINDOW_FRAMES),
    "Cheating object detected": deque([False]*WINDOW_FRAMES, maxlen=WINDOW_FRAMES),
}

# --- CHEATING CHECKS (MODULARIZED) ---

def check_person_count(num_persons, indicators):
    if num_persons == 0:
        indicators.append("No person detected")
        return True
    if num_persons > 1:
        indicators.append("Multiple persons detected (YOLO)")
        return True
    return False

def check_cheating_objects(cheating_objects_boxes, indicators):
    if len(cheating_objects_boxes) > 0:
        indicators.append("Cheating object detected")
        return True
    return False

# --- MAIN LOOP ---

cap = cv2.VideoCapture(1)
frame_count = 0
presence_history = deque(maxlen=PRESENCE_HISTORY_SIZE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue  # Skip frame for efficiency

    cheating_indicators = []

    height, width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- YOLO OBJECT DETECTION ---
    results = yolo(frame)
    det = results[0]
    boxes = det.boxes.xyxy.cpu().numpy() if det.boxes.xyxy is not None else np.array([])
    classes = det.boxes.cls.cpu().numpy() if det.boxes.cls is not None else np.array([])
    confidences = det.boxes.conf.cpu().numpy() if det.boxes.conf is not None else np.array([])

    # Filter by class and confidence
    person_boxes = boxes[(classes == 0) & (confidences > CONFIDENCE_THRESHOLD)] if len(boxes) else []
    cheating_objects_boxes = boxes[
        np.isin(classes, cheating_classes) & (confidences > CONFIDENCE_THRESHOLD)
    ] if len(boxes) else []
    num_persons = len(person_boxes)

    # --- PRESENCE SMOOTHING ---
    presence_history.append(num_persons)
    smoothed_persons = round(np.median(presence_history))

    # --- MODULARIZED CHEATING CHECKS ---
    detected_person = check_person_count(smoothed_persons, cheating_indicators)
    detected_cheating = check_cheating_objects(cheating_objects_boxes, cheating_indicators)

    # --- TEMPORAL WINDOW LOGIC (3s, 35%) ---
    confirmed_cheating = False
    confirmed_indicators = []
    for indicator in indicator_windows:
        detected = indicator in cheating_indicators
        indicator_windows[indicator].append(detected)
        presence_ratio = sum(indicator_windows[indicator]) / WINDOW_FRAMES
        # Uncomment for debugging:
        # print(f"{indicator}: {presence_ratio*100:.1f}% in last {WINDOW_SECONDS}s")
        if presence_ratio >= THRESHOLD_RATIO:
            confirmed_cheating = True
            confirmed_indicators.append(indicator)

    # --- DRAW INDICATORS ---
    y_disp = 50
    for indicator in confirmed_indicators:
        cv2.putText(frame, indicator, (50, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_disp += 30

    # --- DRAW BOUNDING BOXES ---
    for i in range(len(boxes)):
        if confidences[i] > CONFIDENCE_THRESHOLD:
            box = boxes[i]
            cls = int(classes[i])
            x1, y1, x2, y2 = map(int, box)
            if cls == 0:
                color = (0, 255, 0)
            elif cls in cheating_classes:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = class_names.get(cls, f"class {cls}")
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"Persons: {smoothed_persons}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- Prepare detection metadata for logging ---
    detection_metadata = {
        "detected_classes": [],
        "confidences": [],
        "boxes": [],
    }
    for i in range(len(boxes)):
        if confidences[i] > CONFIDENCE_THRESHOLD:
            detection_metadata["detected_classes"].append(class_names.get(int(classes[i]), str(int(classes[i]))))
            detection_metadata["confidences"].append(float(confidences[i]))
            detection_metadata["boxes"].append([int(coord) for coord in boxes[i]])

    # --- LOGGING & EVIDENCE IMAGE & ALARM ---
    if confirmed_cheating:
        for indicator in confirmed_indicators:
            if should_trigger(indicator):
                log_event(indicator, frame, detection_metadata)
                trigger_alarm()
                # Optionally: send_frame_to_server(frame, indicator)

    cv2.imshow("Cheating Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
mp_pose.close()