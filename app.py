"""
Flask REST API for Fire and Smoke Detection
Provides HTTP endpoints for inference
"""

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
from fire_smoke_detection.inference.predictor import FireSmokeDetector


# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Initialize detector
detector = FireSmokeDetector(
    model_path="artifacts/model_training/best.pt",
    conf_threshold=0.25
)

# Create folders
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "css").mkdir(exist_ok=True, parents=True)
(STATIC_DIR / "js").mkdir(exist_ok=True, parents=True)


@app.route('/', methods=['GET'])
def index():
    """Serve the main web interface"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'fire_smoke_detection',
        'version': '1.0.0'
    })


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    info = detector.get_model_info()
    return jsonify(info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict on uploaded image
    
    Request:
        - file: Image file (multipart/form-data)
        - conf: Confidence threshold (optional, default: 0.25)
    
    Response:
        - detections: List of detections
        - num_detections: Number of detections
        - image_size: Image dimensions
    """
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Get confidence threshold
    conf = float(request.form.get('conf', 0.25))
    detector.conf_threshold = conf
    
    # Save uploaded file
    file_path = UPLOAD_FOLDER / file.filename
    file.save(file_path)
    
    try:
        # Run prediction
        result = detector.predict_image(file_path, save=False, return_base64=True)
        
        img = Image.open(file_path)
        image_size = {'width': img.width, 'height': img.height}
        
        # Clean up
        file_path.unlink()
        
        return jsonify({
            'detections': result['detections'],
            'num_detections': result['num_detections'],
            'image_base64': result.get('image_base64'),
            'image_size': image_size,
            'status': 'success'
        })
    
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/predict/base64', methods=['POST'])
def predict_base64():
    """
    Predict on base64 encoded image
    
    Request (JSON):
        - image: Base64 encoded image string
        - conf: Confidence threshold (optional, default: 0.25)
    
    Response:
        - detections: List of detections
        - num_detections: Number of detections
    """
    data = request.get_json()
    
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get confidence threshold
    conf = float(data.get('conf', 0.25))
    detector.conf_threshold = conf
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data))
        
        # Save temporarily
        temp_path = UPLOAD_FOLDER / 'temp.jpg'
        image.save(temp_path)
        
        # Run prediction
        result = detector.predict_image(temp_path, save=False)
        
        # Clean up
        temp_path.unlink()
        
        return jsonify({
            'detections': result['detections'],
            'num_detections': result['num_detections']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict on multiple images
    
    Request:
        - files: Multiple image files (multipart/form-data)
        - conf: Confidence threshold (optional, default: 0.25)
    
    Response:
        - results: List of prediction results
        - total_images: Number of images processed
        - total_detections: Total number of detections
    """
    # Check if files are present
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if len(files) == 0:
        return jsonify({'error': 'No files provided'}), 400
    
    # Get confidence threshold
    conf = float(request.form.get('conf', 0.25))
    detector.conf_threshold = conf
    
    try:
        results = []
        total_detections = 0
        
        for file in files:
            # Save file
            file_path = UPLOAD_FOLDER / file.filename
            file.save(file_path)
            
            # Run prediction
            result = detector.predict_image(file_path, save=False)
            results.append({
                'filename': file.filename,
                'detections': result['detections'],
                'num_detections': result['num_detections']
            })
            
            total_detections += result['num_detections']
            
            # Clean up
            file_path.unlink()
        
        return jsonify({
            'results': results,
            'total_images': len(results),
            'total_detections': total_detections
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Fire and Smoke Detection API...")
    print("Model loaded successfully")
    print("API running on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /model/info - Model information")
    print("  POST /predict - Predict on single image")
    print("  POST /predict/base64 - Predict on base64 image")
    print("  POST /predict/batch - Predict on multiple images")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
