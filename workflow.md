## Workflow of the project
1. Set Up Project Dependencies
Populate requirements.txt with necessary libraries (PyTorch/TensorFlow, OpenCV, YOLO, etc.)
Create virtual environment and install dependencies
2. Configure Project Files
config/config.yaml: Define paths, model parameters, data locations
params.yaml: Set hyperparameters (learning rate, batch size, epochs, etc.)
schema.yaml 
: Define data validation schema
3. Implement Core Components (in src/fire_smoke_detection/components/)
Data Ingestion: Load and organize your dataset
Data Validation: Verify data integrity and format
Data Transformation: Preprocessing, augmentation, train-test split
Model Training: Implement training logic (likely YOLO-based for object detection)
Model Evaluation: Metrics calculation (mAP, precision, recall, F1)
4. Build Utility Functions
Common helper functions in utils/common.py
Configuration management in config/configuration.py
5. Create Training Pipeline
End-to-end pipeline in pipeline/ directory
Orchestrate all components
6. Train and Evaluate Model
Run training on your dataset
Evaluate performance
Save best model weights
7. Build Inference & Deployment
Create prediction pipeline
Build web interface (Flask/FastAPI)
Containerize with Docker