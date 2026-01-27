# Fire and Smoke Detection - Deployment Guide

## 🚀 Quick Start

### Option 1: Command-Line Inference

```bash
# Detect on single image
python3 predict.py --image path/to/image.jpg

# Detect on video
python3 predict.py --video path/to/video.mp4

# Detect from webcam
python3 predict.py --webcam

# Detect on folder
python3 predict.py --folder path/to/images/
```

### Option 2: REST API

```bash
# Start API server
python3 app.py

# API will be available at http://localhost:5000
```

### Option 3: Python API

```python
from fire_smoke_detection.inference.predictor import FireSmokeDetector

# Initialize detector
detector = FireSmokeDetector(model_path="artifacts/model_training/best.pt")

# Predict on image
result = detector.predict_image("image.jpg")
print(result['detections'])
```

---

## 📦 Installation

### Install Deployment Dependencies

```bash
pip install -r requirements_deploy.txt
```

**Dependencies:**
- ultralytics (YOLOv8)
- opencv-python (Image/video processing)
- flask (REST API)
- flask-cors (CORS support)
- gunicorn (Production server)

---

## 🔧 Command-Line Usage

### Basic Commands

```bash
# Single image
python3 predict.py --image test.jpg

# Video file
python3 predict.py --video test.mp4

# Webcam (press 'q' to quit)
python3 predict.py --webcam

# Folder of images
python3 predict.py --folder test_images/
```

### Advanced Options

```bash
# Adjust confidence threshold
python3 predict.py --image test.jpg --conf 0.5

# Use CPU instead of GPU
python3 predict.py --image test.jpg --device cpu

# Custom model path
python3 predict.py --image test.jpg --model path/to/model.pt

# Don't save results
python3 predict.py --image test.jpg --no-save

# Custom output directory
python3 predict.py --image test.jpg --output results/

# Show results (images only)
python3 predict.py --image test.jpg --show

# Use specific camera
python3 predict.py --webcam --camera 1
```

---

## 🌐 REST API

### Start API Server

```bash
# Development server
python3 app.py

# Production server (with gunicorn)
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 app:app
```

### API Endpoints

#### 1. Health Check

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "fire_smoke_detection",
  "version": "1.0.0"
}
```

#### 2. Model Information

```bash
curl http://localhost:5000/model/info
```

**Response:**
```json
{
  "model_path": "artifacts/model_training/best.pt",
  "classes": {"0": "fire", "1": "both"},
  "num_classes": 2,
  "conf_threshold": 0.25,
  "iou_threshold": 0.45,
  "device": "0"
}
```

#### 3. Predict on Image

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
  "detections": [
    {
      "class": "fire",
      "class_id": 0,
      "confidence": 0.92,
      "bbox": {
        "x1": 100.5,
        "y1": 150.2,
        "x2": 300.8,
        "y2": 400.1
      }
    }
  ],
  "num_detections": 1,
  "image_size": {
    "width": 640,
    "height": 480
  }
}
```

#### 4. Predict with Custom Confidence

```bash
curl -X POST -F "file=@image.jpg" -F "conf=0.5" http://localhost:5000/predict
```

#### 5. Predict on Base64 Image

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_string", "conf": 0.25}' \
  http://localhost:5000/predict/base64
```

#### 6. Batch Prediction

```bash
curl -X POST -F "files=@image1.jpg" -F "files=@image2.jpg" \
  http://localhost:5000/predict/batch
```

**Response:**
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "detections": [...],
      "num_detections": 2
    },
    {
      "filename": "image2.jpg",
      "detections": [...],
      "num_detections": 1
    }
  ],
  "total_images": 2,
  "total_detections": 3
}
```

---

## 🐍 Python API

### Basic Usage

```python
from fire_smoke_detection.inference.predictor import FireSmokeDetector

# Initialize detector
detector = FireSmokeDetector(
    model_path="artifacts/model_training/best.pt",
    conf_threshold=0.25,
    device="0"  # "0" for GPU, "cpu" for CPU
)

# Predict on single image
result = detector.predict_image("image.jpg", save=True)

# Print detections
for det in result['detections']:
    print(f"{det['class']}: {det['confidence']:.2%}")
```

### Advanced Usage

```python
# Predict on multiple images
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = detector.predict_batch(image_paths, save=True)

# Predict on video
result = detector.predict_video("video.mp4", save=True)
print(f"Processed {result['frames_processed']} frames")

# Predict on folder
results = detector.predict_folder("test_images/", save=True)
print(f"Processed {len(results)} images")

# Real-time webcam detection
detector.predict_webcam(camera_index=0)

# Get model information
info = detector.get_model_info()
print(f"Classes: {info['classes']}")
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t fire-smoke-detection .
```

### Run Docker Container

```bash
docker run -p 5000:5000 fire-smoke-detection
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./artifacts:/app/artifacts
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

---

## ☁️ Cloud Deployment

### AWS (EC2)

1. Launch EC2 instance (Ubuntu 22.04)
2. Install Docker
3. Clone repository
4. Build and run Docker container
5. Configure security group (port 5000)

### Google Cloud (Cloud Run)

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/fire-smoke-detection

# Deploy to Cloud Run
gcloud run deploy fire-smoke-detection \
  --image gcr.io/PROJECT_ID/fire-smoke-detection \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure (Container Instances)

```bash
# Build and push to ACR
az acr build --registry myregistry --image fire-smoke-detection .

# Deploy to ACI
az container create \
  --resource-group myResourceGroup \
  --name fire-smoke-detection \
  --image myregistry.azurecr.io/fire-smoke-detection \
  --dns-name-label fire-smoke-detection \
  --ports 5000
```

---

## 📱 Mobile Deployment

### Export to ONNX

```python
from ultralytics import YOLO

model = YOLO('artifacts/model_training/best.pt')
model.export(format='onnx')
```

### Export to TensorFlow Lite

```python
model.export(format='tflite')
```

### Export to TensorRT (NVIDIA)

```python
model.export(format='engine')
```

---

## 🔒 Production Best Practices

### 1. Use Production Server

```bash
# Don't use Flask development server in production
# Use gunicorn instead
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 app:app
```

### 2. Add Authentication

```python
# Add API key authentication
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your-secret-key':
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ...
```

### 3. Add Rate Limiting

```bash
pip install flask-limiter
```

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ...
```

### 4. Add Logging

```python
import logging

logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 5. Monitor Performance

```python
import time

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - request.start_time
    logging.info(f"Request took {duration:.2f}s")
    return response
```

---

## 📊 Performance Optimization

### GPU Acceleration

```python
# Use GPU for faster inference
detector = FireSmokeDetector(device="0")  # GPU 0
```

### Batch Processing

```python
# Process multiple images at once
results = detector.predict_batch(image_paths)
```

### Model Optimization

```python
# Export to TensorRT for NVIDIA GPUs
model.export(format='engine')
```

---

## 🧪 Testing

### Test Command-Line

```bash
# Test on sample image
python3 predict.py --image artifacts/data_transformation/test/images/img_105_jpg.rf.a7dfff51279e0f7c3594055d005bea41.jpg
```

### Test API

```bash
# Start API
python3 app.py &

# Test health endpoint
curl http://localhost:5000/health

# Test prediction
curl -X POST -F "file=@test.jpg" http://localhost:5000/predict
```

### Test Python API

```python
from fire_smoke_detection.inference.predictor import FireSmokeDetector

detector = FireSmokeDetector()
result = detector.predict_image("test.jpg")
assert result['num_detections'] >= 0
print("✅ Test passed!")
```

---

## 📚 API Client Examples

### Python Client

```python
import requests

# Predict on image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'file': f}
    )

result = response.json()
print(f"Detections: {result['num_detections']}")
```

### JavaScript Client

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:5000/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

### cURL Client

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

---

## 🎯 Use Cases

### 1. Real-time Monitoring

```python
# Monitor webcam for fire/smoke
detector.predict_webcam(camera_index=0)
```

### 2. Video Analysis

```python
# Analyze security footage
result = detector.predict_video("security_footage.mp4")
```

### 3. Batch Processing

```python
# Process folder of images
results = detector.predict_folder("surveillance_images/")
```

### 4. Integration with Alarm System

```python
result = detector.predict_image("camera_feed.jpg")

if result['num_detections'] > 0:
    # Trigger alarm
    send_alert("Fire/Smoke detected!")
```

---

## 📞 Quick Reference

```bash
# Command-line
python3 predict.py --image test.jpg
python3 predict.py --video test.mp4
python3 predict.py --webcam

# API
python3 app.py
curl -X POST -F "file=@test.jpg" http://localhost:5000/predict

# Docker
docker build -t fire-smoke-detection .
docker run -p 5000:5000 fire-smoke-detection
```

**Model Performance:**
- mAP@0.5: 95.24%
- Inference Speed: ~238 FPS (GPU)
- Model Size: 6.0 MB

**Status:** ✅ Production Ready
