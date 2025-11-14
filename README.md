# 🌊 Astronaut in the Ocean - Object Detection Web App

A web application for object detection in ocean imagery using YOLOv8 and FastAPI.

## Features

- **YOLOv8 Integration**: Uses the state-of-the-art YOLOv8 model for object detection
- **FastAPI Backend**: Fast, modern Python web framework with automatic API documentation
- **Web Interface**: Simple HTML interface for uploading and testing images
- **REST API**: Easy-to-use API endpoints for programmatic access
- **Real-time Detection**: Fast inference with YOLOv8 nano model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HemdatShefer/astronaut-in-the-ocean.git
cd astronaut-in-the-ocean
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the FastAPI server:
```bash
python app.py
```

Or use uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at:
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

### Using the Web Interface

1. Open http://localhost:8000 in your browser
2. Click to select an image file
3. Click "Detect Objects" to run detection
4. View the results showing detected objects, confidence scores, and bounding boxes

### Using the API

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Detect Objects in an Image
```bash
curl -X POST "http://localhost:8000/detect/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

#### Get Model Information
```bash
curl http://localhost:8000/model/info
```

### API Response Format

The `/detect/` endpoint returns JSON with the following structure:
```json
{
  "filename": "image.jpg",
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "box": [100.5, 200.3, 300.7, 450.2]
    }
  ],
  "count": 1,
  "inference_time": 0.123,
  "image_shape": [720, 1280, 3]
}
```

## API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check endpoint
- `POST /detect/` - Upload and detect objects in an image
- `GET /model/info` - Get information about the loaded model
- `GET /docs` - Interactive API documentation (Swagger UI)

## Model

The application uses YOLOv8n (nano) model by default, which is automatically downloaded on first run. This model provides:
- Fast inference speed
- Good accuracy for general object detection
- Support for 80 COCO dataset classes

You can modify `MODEL_PATH` in `app.py` to use different YOLOv8 variants:
- `yolov8n.pt` - Nano (fastest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate)

## Technologies Used

- **FastAPI**: Modern, fast web framework for building APIs
- **Ultralytics YOLOv8**: State-of-the-art object detection model
- **OpenCV**: Image processing
- **Uvicorn**: ASGI server for FastAPI
- **Pillow**: Python Imaging Library

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
