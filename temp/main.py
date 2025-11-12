from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Run detection
results = model("temp.jpg")

print("AAA")
print(results[0])
print("BBB")
# Show and print info
for r in results:
    r.show()  # show annotated image
    for box in r.boxes:
        cls = model.names[int(box.cls)]      # class name
        conf = float(box.conf)               # confidence
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # coordinates
        # print(f"{cls}: {conf:.2f}  [{x1}, {y1}, {x2}, {y2}]")
