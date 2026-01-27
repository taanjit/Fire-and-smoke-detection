# Fire and Smoke Detection

A deep learning-based fire and smoke detection system using YOLOv8 for real-time object detection.

## Features

- Real-time fire and smoke detection
- YOLOv8-based object detection
- Comprehensive data validation pipeline
- Model training and evaluation
- Web-based deployment ready

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Data Validation
```bash
python main.py
```

## Project Structure

```
Fire_smoke_detection/
├── data/                  # Dataset directory
├── src/                   # Source code
│   └── fire_smoke_detection/
│       ├── components/    # Pipeline components
│       ├── config/        # Configuration management
│       ├── entity/        # Data entities
│       ├── pipeline/      # Training pipelines
│       └── utils/         # Utility functions
├── config/                # Configuration files
├── artifacts/             # Training artifacts
└── main.py               # Main execution script
```

## License

MIT License
