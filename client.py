import cv2
import requests

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Encode frame as JPEG
    _, frame_jpg = cv2.imencode('.jpg', frame)
    files = {'file': ('frame.jpg', frame_jpg.tobytes(), 'image/jpeg')}

    # Send to FastAPI backend
    response = requests.post('http://localhost:8000/detect-frame/', files=files)
    print(response.json())

    # Show webcam and break with 'q'
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 