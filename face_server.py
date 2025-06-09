import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import face_recognition
import shutil
from datetime import datetime

REFERENCE_DIR = "reference_photos"
INTRUDER_DIR = "intruders"
LOG_FILE = "intruder_log.csv"
os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(INTRUDER_DIR, exist_ok=True)

app = FastAPI()

def get_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None

def log_intruder(reference_path, suspect_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp},{reference_path},{suspect_path}\n")

@app.post("/compare-faces/")
async def compare_faces(
    reference: UploadFile = File(...), 
    suspect: UploadFile = File(...),
    session_id: str = Form("default")
):
    # Save the uploaded reference and suspect images temporarily
    ref_path = os.path.join(REFERENCE_DIR, f"{session_id}_ref_{reference.filename}")
    suspect_path = os.path.join(INTRUDER_DIR, f"{session_id}_suspect_{suspect.filename}")
    with open(ref_path, "wb") as f:
        shutil.copyfileobj(reference.file, f)
    with open(suspect_path, "wb") as f:
        shutil.copyfileobj(suspect.file, f)

    # Perform face encoding
    ref_encoding = get_encoding(ref_path)
    suspect_encoding = get_encoding(suspect_path)
    if ref_encoding is None or suspect_encoding is None:
        os.remove(ref_path)
        os.remove(suspect_path)
        return JSONResponse(
            {"result": "fail", "reason": "Could not find a face in one or both images."}, status_code=400
        )
    is_same = face_recognition.compare_faces([ref_encoding], suspect_encoding, tolerance=0.6)[0]
    if is_same:
        # Clean up suspect image if same
        os.remove(suspect_path)
        result = {"result": "success", "match": True, "message": "Same person."}
    else:
        # Log the incident
        log_intruder(ref_path, suspect_path)
        result = {
            "result": "success",
            "match": False,
            "message": f"Different person detected! Intruder saved at {suspect_path}",
        }
    return result