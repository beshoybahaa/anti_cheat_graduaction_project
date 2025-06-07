from ultralytics import YOLO

# Download and load the YOLOv8n model
model = YOLO('yolov8n.pt')

# Export the model to ONNX format
model.export(format='onnx') 