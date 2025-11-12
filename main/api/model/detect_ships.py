"""
Simple function to detect ships - just call it and get results!

Usage:
    from detect_ships import detect_ships
    
    results = detect_ships("image.jpg")
    print(results)
"""

import cv2
import torch
import clip
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Global detector instance (loaded once)
_detector = None
_yolo = None
_clip_model = None
_clip_preprocess = None
_text_features = None
_device = None
_category_names = None


def _initialize():
    """Initialize models (called automatically first time)"""
    global _detector, _yolo, _clip_model, _clip_preprocess, _text_features, _device, _category_names
    
    if _detector is not None:
        return  # Already initialized
    
    print("Loading models (first time only)...")
    
    # Load YOLO
    _yolo = YOLO("yolo12n.pt")
    
    # Load CLIP
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _clip_model, _clip_preprocess = clip.load("ViT-B/16", device=_device)
    _clip_model.eval()
    
    # Ship categories
    ship_types = [
        "a photo of a fishing boat on the ocean",
        "a photo of a cargo ship on the ocean",
        "a photo of a large cargo container ship",
        "a photo of a kayak on the water",
        "a photo of a submarine in the water",
        "a photo of a buoy floating in the ocean",
        "a photo of a small patrol boat",
        "a photo of a tugboat",
        "a photo of a large tanker ship"
    ]
    
    _category_names = [
        "fishing boat", "cargo ship", "container ship", "kayak",
        "submarine", "buoy", "patrol boat", "tugboat", "tanker ship"
    ]
    
    # Encode text
    with torch.no_grad():
        text_tokens = clip.tokenize(ship_types).to(_device)
        _text_features = _clip_model.encode_text(text_tokens)
        _text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)
    
    _detector = True  # Mark as initialized
    print(f"Models ready! Device: {_device}\n")


def detect_ships(img, conf_threshold=0.1):
    """
    Detect and classify ships in an image
    
    Args:
        image: Can be one of:
               - str: Path to image file (e.g., "ship.jpg")
               - bytes: Raw image bytes
               - numpy.ndarray: OpenCV image (cv2 image)
        conf_threshold: Detection confidence threshold (default: 0.1)
    
    Returns:
        Dictionary with results:
        {
            'detections': [
                {
                    'bbox': [x1, y1, x2, y2],
                    'class': 'boat',
                    'subclass': 'fishing boat',
                    'confidence': 0.95
                },
                ...
            ],
            'summary': {
                'total': 7,
                'boats': 6,
                'people': 1
            }
        }
    
    Examples:
        # From file path
        results = detect_ships("ship.jpg")
        
        # From bytes
        with open("ship.jpg", "rb") as f:
            image_bytes = f.read()
        results = detect_ships(image_bytes)
        
        # From cv2 image
        import cv2
        img = cv2.imread("ship.jpg")
        results = detect_ships(img)
    """
    # Initialize models if needed
    _initialize()
    
    # Load image based on input type

    img_height, img_width = img.shape[:2]
    
    # Stage 1: YOLO Detection
    results = _yolo(img, conf=conf_threshold, verbose=False)[0]
    
    detections = []
    boat_count = 0
    people_count = 0
    
    # Stage 2: Process each detection
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = _yolo.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detection_conf = float(box.conf[0])
        
        # Bounds check
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)
        
        if cls_name == 'boat':
            # Classify boat type
            crop = img[y1:y2, x1:x2]
            if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)
                
                # CLIP classification
                img_input = _clip_preprocess(crop_pil).unsqueeze(0).to(_device)
                with torch.no_grad():
                    image_features = _clip_model.encode_image(img_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ _text_features.T)
                    probs = similarity.softmax(dim=-1).squeeze(0)
                    
                    best_idx = probs.argmax().item()
                    best_prob = probs[best_idx].item()
                    
                    subclass = _category_names[best_idx]
                    confidence = best_prob
            else:
                subclass = "boat"
                confidence = detection_conf
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class': 'boat',
                'subclass': subclass,
                'confidence': confidence
            })
            boat_count += 1
        
        elif cls_name == 'person':
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class': 'person',
                'subclass': 'swimmer/person',
                'confidence': detection_conf
            })
            people_count += 1
    
    # Return as dictionary
    return {
        'detections': detections,
        'summary': {
            'total': len(detections),
            'boats': boat_count,
            'people': people_count
        }
    }