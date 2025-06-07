import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import time
import csv
from datetime import datetime
from collections import deque
import os

app = FastAPI()

YOLO_MODEL = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.6
ALARM_COOLDOWN = 10
FRAME_SKIP = 2
PRESENCE_HISTORY_SIZE = 7

class_names = {0: "person", 67: "cell phone", 73: "book"}
cheating_classes = [67, 73]
ALARM_SOUND = False  # Set True if your server can play sound
last_alarm_time = 0

# In-memory state for each client/session (use DB or Redis for production)
sessions = {}

# --- Logging function ---
def log_event(session_id, indicator, detection_metadata):
    ts = int(time.time())
    iso_timestamp = datetime.utcnow().isoformat()
    filename = f"cheating_{session_id}_{ts}.jpg"
    # Save image if present
    if detection_metadata.get("frame") is not None:
        cv2.imwrite(filename, detection_metadata["frame"])
    # CSV log
    csv_name = f"cheating_log_{session_id}.csv"
    with open(csv_name, "a", newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "unix_timestamp", "iso_timestamp", "indicator", 
                "filename", "detected_classes", "confidences", "boxes"
            ])
        writer.writerow([
            ts, iso_timestamp, indicator, filename,
            "|".join(map(str, detection_metadata.get("detected_classes", []))),
            "|".join("{:.3f}".format(c) for c in detection_metadata.get("confidences", [])),
            "|".join(str(b) for b in detection_metadata.get("boxes", []))
        ])

def trigger_alarm():
    global last_alarm_time
    if ALARM_SOUND and (time.time() - last_alarm_time > ALARM_COOLDOWN):
        try:
            os.system('say "Cheating detected!"')
        except Exception as e:
            print("Alarm error:", e)
        last_alarm_time = time.time()

# --- Cheating Checks ---
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

# --- Session State Management ---
def get_or_init_session(session_id):
    if session_id not in sessions:
        FPS_ESTIMATE = 10
        WINDOW_SECONDS = 3
        WINDOW_FRAMES = WINDOW_SECONDS * FPS_ESTIMATE
        THRESHOLD_RATIO = 0.35
        sessions[session_id] = {
            "frame_count": 0,
            "presence_history": deque(maxlen=PRESENCE_HISTORY_SIZE),
            "indicator_windows": {
                "No person detected": deque([False]*WINDOW_FRAMES, maxlen=WINDOW_FRAMES),
                "Multiple persons detected (YOLO)": deque([False]*WINDOW_FRAMES, maxlen=WINDOW_FRAMES),
                "Cheating object detected": deque([False]*WINDOW_FRAMES, maxlen=WINDOW_FRAMES),
            },
            "WINDOW_FRAMES": WINDOW_FRAMES,
            "THRESHOLD_RATIO": THRESHOLD_RATIO,
        }
    return sessions[session_id]

# --- Load YOLO once ---
yolo = YOLO(YOLO_MODEL)

@app.post("/detect-frame/")
async def detect_frame(file: UploadFile = File(...), session_id: str = Form(...)):
    # Get/init session state
    state = get_or_init_session(session_id)
    state["frame_count"] += 1
    if state["frame_count"] % FRAME_SKIP != 0:
        return {"skipped": True}

    # Read image file bytes
    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    cheating_indicators = []
    height, width = frame.shape[:2]

    # --- YOLO detection ---
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

    # --- Presence smoothing ---
    state["presence_history"].append(num_persons)
    smoothed_persons = round(np.median(state["presence_history"]))

    # --- Modularized checks ---
    check_person_count(smoothed_persons, cheating_indicators)
    check_cheating_objects(cheating_objects_boxes, cheating_indicators)

    # --- Temporal window logic ---
    confirmed_cheating = False
    confirmed_indicators = []
    for indicator in state["indicator_windows"]:
        detected = indicator in cheating_indicators
        state["indicator_windows"][indicator].append(detected)
        presence_ratio = sum(state["indicator_windows"][indicator]) / state["WINDOW_FRAMES"]
        if presence_ratio >= state["THRESHOLD_RATIO"]:
            confirmed_cheating = True
            confirmed_indicators.append(indicator)

    # --- Metadata for log ---
    detection_metadata = {
        "detected_classes": [class_names.get(int(classes[i]), str(int(classes[i]))) for i in range(len(boxes)) if confidences[i] > CONFIDENCE_THRESHOLD],
        "confidences": [float(confidences[i]) for i in range(len(boxes)) if confidences[i] > CONFIDENCE_THRESHOLD],
        "boxes": [[int(coord) for coord in boxes[i]] for i in range(len(boxes)) if confidences[i] > CONFIDENCE_THRESHOLD],
        "frame": frame
    }

    # --- Logging & Alarm ---
    if confirmed_cheating:
        for indicator in confirmed_indicators:
            log_event(session_id, indicator, detection_metadata)
            if ALARM_SOUND:
                trigger_alarm()

    return {
        "cheating_detected": confirmed_cheating,
        "confirmed_indicators": confirmed_indicators,
        "persons": int(smoothed_persons),
        "cheating_classes": detection_metadata["detected_classes"],
        "confidences": detection_metadata["confidences"]
    }
