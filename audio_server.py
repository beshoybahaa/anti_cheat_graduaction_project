import os
import numpy as np
import torch
import time
import csv
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form
from scipy.io import wavfile

model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
(get_speech_timestamps, _, read_audio, *_) = utils

RATE = 16000
WINDOW_SECONDS = 3
MIN_SEGMENT_SECONDS = 0.2 
MIN_SEGMENTS = 1
NOTIFICATION_COOLDOWN = 10
SILERO_THRESHOLD = 0.2     # Very sensitive

AUDIO_LOG_DIR = "audio_logs"
os.makedirs(AUDIO_LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(AUDIO_LOG_DIR, "cheating_log.csv")

app = FastAPI()
last_notification = {}

def log_event(session_id, indicator, audio_segment, detection_metadata=None):
    ts = int(time.time())
    iso_timestamp = datetime.now(timezone.utc).isoformat()
    wav_filename = os.path.join(AUDIO_LOG_DIR, f"cheating_{session_id}_{ts}.wav")
    wavfile.write(wav_filename, RATE, (audio_segment * 32767).astype(np.int16))
    with open(LOG_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "unix_timestamp", "iso_timestamp", "session_id", "indicator", 
                "filename", "segment_start", "segment_end"
            ])
        segment_start = detection_metadata.get("segment_start", "") if detection_metadata else ""
        segment_end = detection_metadata.get("segment_end", "") if detection_metadata else ""
        writer.writerow([
            ts, iso_timestamp, session_id, indicator, wav_filename, segment_start, segment_end
        ])
    return wav_filename

@app.post("/analyze-audio/")
async def analyze_audio(
    file: UploadFile = File(...),
    session_id: str = Form("default")
):
    contents = await file.read()
    try:
        rate, audio_np = wavfile.read(file.file)
        audio_np = audio_np.astype(np.float32) / 32768.0
    except Exception:
        audio_np = np.frombuffer(contents, dtype=np.int16).astype(np.float32) / 32768.0
        rate = RATE

    if len(audio_np) > rate * WINDOW_SECONDS:
        audio_np = audio_np[-int(rate * WINDOW_SECONDS):]

    audio_tensor = torch.from_numpy(audio_np)
    timestamps = get_speech_timestamps(
        audio_tensor, model,
        sampling_rate=rate,
        min_speech_duration_ms=int(MIN_SEGMENT_SECONDS * 1000),
        threshold=SILERO_THRESHOLD
    )

    now = time.time()
    last_notif = last_notification.get(session_id, 0)
    cheating_detected = False
    meta = {}
    log_name = None

    if len(timestamps) >= MIN_SEGMENTS and now - last_notif > NOTIFICATION_COOLDOWN:
        cheating_detected = True
        ts_data = timestamps[0]
        log_name = log_event(
            session_id=session_id,
            indicator="speech_detected",
            audio_segment=audio_np,
            detection_metadata={
                "segment_start": float(ts_data['start']) / float(rate),
                "segment_end": float(ts_data['end']) / float(rate),
            }
        )
        last_notification[session_id] = now
        meta = {
            "segment_start": float(ts_data['start']) / float(rate),
            "segment_end": float(ts_data['end']) / float(rate),
            "log_file": str(log_name)
        }

    return {
        "cheating_detected": bool(cheating_detected),
        "meta": meta,
        "num_segments": int(len(timestamps)),
        "message": (
            "Sustained speech detected and logged." if cheating_detected else "No cheating detected."
        )
    }
