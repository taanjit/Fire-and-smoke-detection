# 🔥 FireGuard AI: Fire & Smoke Detection System

A production-ready, deep learning-based detection system using **YOLOv8**. This project provides a robust pipeline for data validation, model training, evaluation, and an interactive web-based deployment.

[![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)](https://github.com/ultralytics/ultralytics)
[![Accuracy](https://img.shields.io/badge/mAP@0.5-95.2%25-green)]()
[![Flask](https://img.shields.io/badge/Deployed-Flask-blue)]()

## 🌟 Key Features

- **High Precision**: Specially trained model with **95.2% mAP@0.5**.
- **Interactive Web UI**: Modern dark-theme interface with glassmorphism.
- **Real-time Monitoring**: Webcam stream detection directly in the browser.
- **Batch Processing**: Analyze multiple images via REST API or CLI.
- **Robust Pipeline**: Automated data validation, transformation, and training.
- **Fast Inference**: Achieves 230+ FPS on GPU and ~30 FPS on CPU.

## 🚀 Quick Start (Web Interface)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_deploy.txt
   ```
2. **Run the App**:
   ```bash
   python app.py
   ```
3. **Access the UI**: Open [http://localhost:5000](http://localhost:5000) in your browser.

## 🛠️ Project Structure

```text
Fire_smoke_detection/
├── src/fire_smoke_detection/
│   ├── components/    # Training & Evaluation logic
│   ├── pipeline/      # Orchestrated pipeline stages
│   └── inference/     # Predictor API & Webcam support
├── templates/         # HTML Web interface
├── static/            # CSS & JS assets
├── artifacts/         # Trained model (best.pt) & logs
├── config/            # YAML configs (paths, params, schema)
├── app.py             # Flask REST API & Web Server
└── predict.py         # CLI Inference tool
```

## 📊 Model Performance

| Metric | Score |
| :--- | :--- |
| **mAP@0.5** | 95.24% |
| **Precision** | 91.94% |
| **Recall** | 92.86% |
| **F1-Score** | 92.40% |

## 📦 Deployment

The system is container-ready. To build and run with Docker:
```bash
docker build -t fire-detection .
docker run -p 5000:5000 fire-detection
```

## 📜 License
This project is licensed under the MIT License.
