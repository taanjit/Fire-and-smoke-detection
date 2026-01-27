# Utility Functions and Configuration Management

## Overview

This document provides comprehensive documentation for the utility functions and configuration management system in the Fire and Smoke Detection project.

---

## Common Utilities (`utils/common.py`)

### YAML and JSON Operations

#### `read_yaml(path_to_yaml: Path) -> ConfigBox`
Read YAML file and return as ConfigBox object for easy attribute access.

**Example:**
```python
from fire_smoke_detection.utils.common import read_yaml
from pathlib import Path

config = read_yaml(Path("config/config.yaml"))
print(config.artifacts_root)  # Access like attributes
```

#### `write_yaml(path: Path, data: Dict)`
Write dictionary to YAML file with proper formatting.

**Example:**
```python
data = {'model': 'yolov8n', 'epochs': 100}
write_yaml(Path("output.yaml"), data)
```

#### `save_json(path: Path, data: Dict)`
Save dictionary to JSON file with indentation.

#### `load_json(path: Path) -> ConfigBox`
Load JSON file and return as ConfigBox.

---

### Directory and File Operations

#### `create_directories(path_to_directories: list, verbose: bool = True)`
Create multiple directories at once.

**Example:**
```python
create_directories([
    "artifacts/models",
    "artifacts/logs",
    "artifacts/predictions"
])
```

#### `get_size(path: Path) -> str`
Get file size in KB.

**Example:**
```python
size = get_size(Path("model.pt"))
print(size)  # "~ 14523 KB"
```

#### `get_size_mb(path: Path) -> str`
Get file size in MB.

#### `clean_directory(directory: Path, extensions: List[str] = None)`
Remove files from directory, optionally filtering by extension.

**Example:**
```python
# Remove all .tmp and .log files
clean_directory(Path("artifacts"), ['.tmp', '.log'])

# Remove all files
clean_directory(Path("temp"))
```

#### `copy_file(source: Path, destination: Path)`
Copy file with metadata preservation.

---

### Image Processing Utilities

#### `read_image(image_path: Union[str, Path]) -> np.ndarray`
Read image file and return as numpy array.

**Example:**
```python
img = read_image("data/train/images/fire1.jpg")
print(img.shape)  # (640, 640, 3)
```

#### `save_image(image: np.ndarray, save_path: Union[str, Path])`
Save numpy array as image file.

#### `get_image_info(image_path: Union[str, Path]) -> Dict`
Get comprehensive image information.

**Example:**
```python
info = get_image_info("image.jpg")
print(info)
# {
#     'width': 1920,
#     'height': 1080,
#     'channels': 3,
#     'size': '~ 245 KB',
#     'shape': (1080, 1920, 3)
# }
```

#### `resize_image_aspect_ratio(image: np.ndarray, target_size: int) -> np.ndarray`
Resize image maintaining aspect ratio.

**Example:**
```python
img = read_image("large_image.jpg")
resized = resize_image_aspect_ratio(img, 640)
```

---

### System and Performance Utilities

#### `get_timestamp() -> str`
Get current timestamp in YYYYMMDD_HHMMSS format.

**Example:**
```python
timestamp = get_timestamp()
print(timestamp)  # "20260127_153045"
```

#### `calculate_file_hash(file_path: Path, algorithm: str = 'md5') -> str`
Calculate file hash for integrity checking.

**Example:**
```python
hash_val = calculate_file_hash(Path("model.pt"), algorithm='sha256')
```

#### `format_time(seconds: float) -> str`
Format seconds to human-readable time.

**Example:**
```python
print(format_time(45))      # "45.00s"
print(format_time(125))     # "2.08m"
print(format_time(7200))    # "2.00h"
```

#### `count_files(directory: Path, extension: str = None) -> int`
Count files in directory, optionally filtering by extension.

**Example:**
```python
jpg_count = count_files(Path("data/train/images"), ".jpg")
total_count = count_files(Path("data/train/images"))
```

#### `ensure_path_exists(path: Path, is_file: bool = False)`
Ensure path exists, creating directories as needed.

**Example:**
```python
# Ensure directory exists
ensure_path_exists(Path("artifacts/models"))

# Ensure parent directory exists for file
ensure_path_exists(Path("artifacts/models/best.pt"), is_file=True)
```

---

### Data Validation Utilities

#### `validate_file_exists(file_path: Path, file_type: str = "File") -> bool`
Check if file exists with logging.

**Example:**
```python
if validate_file_exists(Path("model.pt"), "Model"):
    # Load model
    pass
```

#### `validate_directory_structure(base_dir: Path, required_dirs: List[str]) -> bool`
Validate that all required subdirectories exist.

**Example:**
```python
required = ['train/images', 'train/labels', 'test/images', 'test/labels']
if validate_directory_structure(Path("data"), required):
    print("Directory structure is valid")
```

---

## Configuration Management (`config/configuration.py`)

### ConfigurationManager Class

Centralized configuration management for all pipeline stages.

#### Initialization

```python
from fire_smoke_detection.config.configuration import ConfigurationManager

config_manager = ConfigurationManager()
```

**Automatically loads:**
- `config/config.yaml` - Paths and directory structure
- `params.yaml` - Hyperparameters and training settings
- `schema.yaml` - Data validation schema

#### Available Methods

##### `get_data_ingestion_config() -> DataIngestionConfig`
Get configuration for data ingestion stage.

##### `get_data_validation_config() -> DataValidationConfig`
Get configuration for data validation stage.

**Example:**
```python
config = config_manager.get_data_validation_config()
print(config.root_dir)
print(config.status_file)
print(config.schema)
```

##### `get_data_transformation_config() -> DataTransformationConfig`
Get configuration for data transformation stage.

##### `get_model_training_config() -> ModelTrainingConfig`
Get configuration for model training stage.

**Example:**
```python
config = config_manager.get_model_training_config()
print(config.params['TRAINING']['epochs'])
print(config.params['MODEL']['variant'])
```

##### `get_model_evaluation_config() -> ModelEvaluationConfig`
Get configuration for model evaluation stage.

---

## Configuration Files

### config.yaml

Defines paths and directory structure for all pipeline stages.

**Structure:**
```yaml
artifacts_root: artifacts

data_validation:
  root_dir: artifacts/data_validation
  status_file: artifacts/data_validation/status.txt
  required_files:
    - train/images
    - train/labels

data_transformation:
  root_dir: artifacts/data_transformation
  image_size: [640, 640]

model_training:
  root_dir: artifacts/model_training
  trained_model_path: artifacts/model_training/best.pt

model_evaluation:
  root_dir: artifacts/model_evaluation
  metrics_file: artifacts/model_evaluation/metrics.json
```

### params.yaml

Defines hyperparameters and training settings.

**Key Sections:**
- `MODEL`: Model architecture and variant
- `TRAINING`: Training hyperparameters
- `AUGMENTATION`: Data augmentation parameters
- `IMAGE`: Image processing settings
- `YOLO`: YOLO-specific parameters

### schema.yaml

Defines data validation rules and constraints.

**Key Sections:**
- `dataset_structure`: Required directories
- `image_files`: Image format and size constraints
- `label_files`: YOLO label format and class definitions
- `quality_checks`: Validation rules

---

## Usage Examples

### Complete Pipeline Example

```python
from fire_smoke_detection.config.configuration import ConfigurationManager
from fire_smoke_detection.components.data_validation import DataValidation
from fire_smoke_detection.components.data_transformation import DataTransformation
from fire_smoke_detection.components.model_training import ModelTraining
from fire_smoke_detection.components.model_evaluation import ModelEvaluation

# Initialize configuration manager
config_manager = ConfigurationManager()

# Data Validation
validation_config = config_manager.get_data_validation_config()
validator = DataValidation(validation_config)
validator.validate_dataset()

# Data Transformation
transform_config = config_manager.get_data_transformation_config()
transformer = DataTransformation(transform_config)
transformer.transform_dataset()

# Model Training
training_config = config_manager.get_model_training_config()
trainer = ModelTraining(training_config)
trainer.train()

# Model Evaluation
eval_config = config_manager.get_model_evaluation_config()
evaluator = ModelEvaluation(eval_config)
metrics = evaluator.evaluate()
```

### Using Utility Functions

```python
from fire_smoke_detection.utils.common import *
from pathlib import Path

# File operations
config = read_yaml(Path("config/config.yaml"))
create_directories(["artifacts/temp", "artifacts/logs"])

# Image processing
img = read_image("data/train/images/fire1.jpg")
info = get_image_info("data/train/images/fire1.jpg")
print(f"Image size: {info['width']}x{info['height']}")

# System utilities
timestamp = get_timestamp()
file_count = count_files(Path("data/train/images"), ".jpg")
print(f"Found {file_count} images")

# Validation
if validate_file_exists(Path("model.pt"), "Model"):
    print("Model found!")
```

---

## Best Practices

### 1. Use ConfigurationManager

Always use ConfigurationManager instead of directly reading config files:

✅ **Good:**
```python
config_manager = ConfigurationManager()
config = config_manager.get_model_training_config()
```

❌ **Bad:**
```python
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)
```

### 2. Use Path Objects

Always use `pathlib.Path` instead of strings for file paths:

✅ **Good:**
```python
from pathlib import Path
model_path = Path("artifacts/model_training/best.pt")
```

❌ **Bad:**
```python
model_path = "artifacts/model_training/best.pt"
```

### 3. Use Utility Functions

Leverage existing utility functions instead of reimplementing:

✅ **Good:**
```python
from fire_smoke_detection.utils.common import read_image
img = read_image("image.jpg")
```

❌ **Bad:**
```python
import cv2
img = cv2.imread("image.jpg")
if img is None:
    raise ValueError("Cannot read image")
```

### 4. Validate Before Processing

Always validate inputs before processing:

```python
if not validate_file_exists(image_path, "Image"):
    return

if not validate_directory_structure(data_dir, required_dirs):
    raise ValueError("Invalid directory structure")
```

### 5. Use Type Annotations

All utility functions use type annotations with `@ensure_annotations`:

```python
@ensure_annotations
def process_image(image_path: Path, size: int) -> np.ndarray:
    # Function implementation
    pass
```

---

## Summary

✅ **Common Utilities**: 25+ helper functions  
✅ **Configuration Management**: Centralized config handling  
✅ **Type Safety**: All functions use type annotations  
✅ **Error Handling**: Comprehensive error checking and logging  
✅ **Documentation**: Full docstrings for all functions  
✅ **Best Practices**: Following Python and ML engineering standards  

**Files:**
- `src/fire_smoke_detection/utils/common.py` - Utility functions
- `src/fire_smoke_detection/config/configuration.py` - Configuration manager
- `config/config.yaml` - Path configurations
- `params.yaml` - Hyperparameters
- `schema.yaml` - Validation schema
