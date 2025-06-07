import pyaudio
import numpy as np
import torch
import time
import queue
import csv
from datetime import datetime, timezone
from scipy.io.wavfile import write as wavwrite

# Load Silero VAD model
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
(get_speech_timestamps, _, read_audio, *_) = utils

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512

WINDOW_SECONDS = 15  # Analyze a longer window
MIN_SEGMENT_SECONDS = 1.5  # Minimum length for a speech segment
MIN_SEGMENTS = 2           # Minimum segments in the window to flag
NOTIFICATION_COOLDOWN = 30 # 1 minute cooldown
HUMAN_BAND = (300, 3400)   # Hz
ENERGY_THRESHOLD = 0.6     # At least 60% energy must be in speech band

def notify_user():
    print("We detected sustained speech, please stay quiet during the quiz.")

def energy_in_band(audio_segment, rate, band):
    fft = np.fft.rfft(audio_segment)
    freqs = np.fft.rfftfreq(len(audio_segment), 1/rate)
    power = np.abs(fft) ** 2
    total_energy = np.sum(power)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    band_energy = np.sum(power[idx])
    return band_energy / total_energy if total_energy > 0 else 0.0

def log_event(indicator, audio_segment, detection_metadata=None):
    ts = int(time.time())
    iso_timestamp = datetime.now(timezone.utc).isoformat()
    filename = f"cheating_{ts}.wav"
    # Save FULL WINDOW audio evidence using scipy
    wavwrite(filename, RATE, (audio_segment * 32767).astype(np.int16))

    with open("cheating_log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "unix_timestamp", "iso_timestamp", "indicator", 
                "filename", "segment_start", "segment_end", "energy_ratio"
            ])
        segment_start = ""
        segment_end = ""
        energy_ratio = ""
        if detection_metadata:
            segment_start = detection_metadata.get("segment_start", "")
            segment_end = detection_metadata.get("segment_end", "")
            energy_ratio = detection_metadata.get("energy_ratio", "")
        writer.writerow([
            ts, iso_timestamp, indicator, filename, segment_start, segment_end, energy_ratio
        ])

def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    buffer = queue.deque(maxlen=int(RATE * WINDOW_SECONDS / CHUNK))
    last_notification = 0

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            buffer.append(audio)
            if len(buffer) == buffer.maxlen:
                audio_np = np.concatenate(buffer)
                audio_tensor = torch.from_numpy(audio_np)
                timestamps = get_speech_timestamps(
                    audio_tensor, model, sampling_rate=RATE, min_speech_duration_ms=int(MIN_SEGMENT_SECONDS * 1000)
                )
                # Only keep segments long enough and with energy in speech band
                long_segments = []
                for ts in timestamps:
                    seg_len = (ts['end'] - ts['start']) / RATE
                    if seg_len >= MIN_SEGMENT_SECONDS:
                        seg_audio = audio_np[ts['start']:ts['end']]
                        band_ratio = energy_in_band(seg_audio, RATE, HUMAN_BAND)
                        if band_ratio >= ENERGY_THRESHOLD:
                            long_segments.append((ts, band_ratio))
                if (
                    len(long_segments) >= MIN_SEGMENTS and
                    time.time() - last_notification > NOTIFICATION_COOLDOWN
                ):
                    notify_user()
                    ts_data, band_ratio = long_segments[0]
                    # Log the FULL WINDOW instead of just the segment!
                    log_event(
                        indicator="speech_detected",
                        audio_segment=audio_np,  # <-- full window
                        detection_metadata={
                            "segment_start": ts_data['start'] / RATE,
                            "segment_end": ts_data['end'] / RATE,
                            "energy_ratio": f"{band_ratio:.3f}"
                        }
                    )
                    last_notification = time.time()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()