"""
FastAPI Object Detection Web Application using YOLOv8
Provides endpoints for detecting objects in images using YOLO model.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
from typing import List, Dict, Any
import os
from contextlib import asynccontextmanager

# Global variable to store the model
model = None
MODEL_PATH = "yolov8n.pt"  # Using YOLOv8 nano model (smallest and fastest)


def load_model():
    """Load the YOLOv8 model. Downloads if not present."""
    global model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model {MODEL_PATH} loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model on startup"""
    load_model()
    yield
    # Cleanup (if needed)


app = FastAPI(
    title="Ocean Object Detection API",
    description="Object detection API using YOLOv8 for ocean imagery",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve a simple HTML interface for testing"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ocean Object Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f0f8ff;
            }
            h1 {
                color: #006994;
                text-align: center;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            input[type="file"] {
                display: block;
                margin: 20px 0;
                padding: 10px;
                border: 2px dashed #006994;
                border-radius: 5px;
                width: 100%;
            }
            button {
                background-color: #006994;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                width: 100%;
            }
            button:hover {
                background-color: #005073;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
                display: none;
            }
            .detection-item {
                padding: 8px;
                margin: 5px 0;
                background-color: #e7f3ff;
                border-left: 4px solid #006994;
                border-radius: 3px;
            }
            #preview {
                max-width: 100%;
                margin-top: 15px;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>🌊 Ocean Object Detection</h1>
        <div class="container">
            <p>Upload an image to detect objects using YOLOv8</p>
            <input type="file" id="imageInput" accept="image/*">
            <img id="preview" style="display:none;">
            <button onclick="detectObjects()">Detect Objects</button>
            <div id="result"></div>
        </div>

        <script>
            const imageInput = document.getElementById('imageInput');
            const preview = document.getElementById('preview');
            
            imageInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    }
                    reader.readAsDataURL(file);
                }
            });

            async function detectObjects() {
                const fileInput = document.getElementById('imageInput');
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files[0]) {
                    alert('Please select an image first');
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    resultDiv.innerHTML = '<p>Processing...</p>';
                    resultDiv.style.display = 'block';
                    
                    const response = await fetch('/detect/', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    
                    if (response.ok) {
                        displayResults(data);
                    } else {
                        resultDiv.innerHTML = `<p style="color: red;">Error: ${data.detail || 'Unknown error'}</p>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            }

            function displayResults(data) {
                const resultDiv = document.getElementById('result');
                let html = `<h3>Detection Results</h3>`;
                html += `<p><strong>Objects found:</strong> ${data.detections.length}</p>`;
                html += `<p><strong>Processing time:</strong> ${data.inference_time.toFixed(3)} seconds</p>`;
                
                if (data.detections.length > 0) {
                    html += '<h4>Detected Objects:</h4>';
                    data.detections.forEach((det, idx) => {
                        html += `
                            <div class="detection-item">
                                <strong>${det.class}</strong> - 
                                Confidence: ${(det.confidence * 100).toFixed(1)}% - 
                                Box: [${det.box.map(b => b.toFixed(1)).join(', ')}]
                            </div>
                        `;
                    });
                } else {
                    html += '<p>No objects detected in the image.</p>';
                }
                
                resultDiv.innerHTML = html;
                resultDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Detect objects in an uploaded image using YOLOv8
    
    Args:
        file: Image file (jpg, png, etc.)
    
    Returns:
        JSON with detected objects, their classes, confidence scores, and bounding boxes
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run inference
        results = model(img)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "box": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        # Get inference time
        inference_time = results[0].speed['inference'] / 1000.0  # Convert to seconds
        
        return {
            "filename": file.filename,
            "detections": detections,
            "count": len(detections),
            "inference_time": inference_time,
            "image_shape": img.shape
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": MODEL_PATH,
        "model_type": "YOLOv8",
        "classes": model.names if hasattr(model, 'names') else []
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
