import os
import numpy as np
import torch
import time
import csv
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from scipy.io import wavfile

# --- Silero VAD Model ---
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
(get_speech_timestamps, _, read_audio, *_) = utils

# --- CONFIG ---
RATE = 16000  # Hz
WINDOW_SECONDS = 3
MIN_SEGMENT_SECONDS = 0.5
MIN_SEGMENTS = 2
NOTIFICATION_COOLDOWN = 10
HUMAN_BAND = (300, 3400)  # Hz
ENERGY_THRESHOLD = 0.6
AUDIO_LOG_DIR = "audio_logs"
os.makedirs(AUDIO_LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(AUDIO_LOG_DIR, "cheating_log.csv")

app = FastAPI()
last_notification = {}

def energy_in_band(audio_segment, rate, band):
    fft = np.fft.rfft(audio_segment)
    freqs = np.fft.rfftfreq(len(audio_segment), 1/rate)
    power = np.abs(fft) ** 2
    total_energy = np.sum(power)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    band_energy = np.sum(power[idx])
    return band_energy / total_energy if total_energy > 0 else 0.0

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
                "filename", "segment_start", "segment_end", "energy_ratio"
            ])
        segment_start = detection_metadata.get("segment_start", "") if detection_metadata else ""
        segment_end = detection_metadata.get("segment_end", "") if detection_metadata else ""
        energy_ratio = detection_metadata.get("energy_ratio", "") if detection_metadata else ""
        writer.writerow([
            ts, iso_timestamp, session_id, indicator, wav_filename, segment_start, segment_end, energy_ratio
        ])
    return wav_filename

@app.post("/analyze-audio/")
async def analyze_audio(
    file: UploadFile = File(...),
    session_id: str = Form("default")
):
    # Read bytes and convert to np array (assume WAV/PCM16)
    contents = await file.read()
    try:
        rate, audio_np = wavfile.read(file.file)  # Try reading as wav
        audio_np = audio_np.astype(np.float32) / 32768.0
    except Exception:
        # Fallback: decode raw bytes as PCM16
        audio_np = np.frombuffer(contents, dtype=np.int16).astype(np.float32) / 32768.0
        rate = RATE

    # Only analyze the last WINDOW_SECONDS
    if len(audio_np) > rate * WINDOW_SECONDS:
        audio_np = audio_np[-int(rate * WINDOW_SECONDS):]

    audio_tensor = torch.from_numpy(audio_np)
    timestamps = get_speech_timestamps(
        audio_tensor, model, sampling_rate=rate, min_speech_duration_ms=int(MIN_SEGMENT_SECONDS * 1000)
    )

    long_segments = []
    for ts in timestamps:
        seg_len = (ts['end'] - ts['start']) / rate
        if seg_len >= MIN_SEGMENT_SECONDS:
            seg_audio = audio_np[ts['start']:ts['end']]
            band_ratio = energy_in_band(seg_audio, rate, HUMAN_BAND)
            if band_ratio >= ENERGY_THRESHOLD:
                long_segments.append((ts, band_ratio))

    # Cooldown check per session
    now = time.time()
    last_notif = last_notification.get(session_id, 0)
    cheating_detected = False
    log_name = None
    meta = {}
    if (
        len(long_segments) >= MIN_SEGMENTS
        and now - last_notif > NOTIFICATION_COOLDOWN
    ):
        cheating_detected = True
        ts_data, band_ratio = long_segments[0]
        log_name = log_event(
            session_id=session_id,
            indicator="speech_detected",
            audio_segment=audio_np,
            detection_metadata={
                "segment_start": ts_data['start'] / rate,
                "segment_end": ts_data['end'] / rate,
                "energy_ratio": f"{band_ratio:.3f}"
            }
        )
        last_notification[session_id] = now
        meta = {
            "segment_start": ts_data['start'] / rate,
            "segment_end": ts_data['end'] / rate,
            "energy_ratio": band_ratio,
            "log_file": log_name
        }

    return {
        "cheating_detected": cheating_detected,
        "meta": meta,
        "num_segments": len(long_segments),
        "message": (
            "Sustained speech detected and logged." if cheating_detected else "No cheating detected."
        )
    }