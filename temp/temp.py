from ultralytics import YOLO

def temp(img):
    model = YOLO("yolov8n.pt")
    results = model(img)
    return results