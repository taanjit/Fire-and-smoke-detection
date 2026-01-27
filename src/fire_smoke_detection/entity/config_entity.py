"""
Configuration Entity Classes
These dataclasses define the structure for various configuration objects
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """Data Ingestion Configuration"""
    root_dir: Path
    source_dir: Path
    train_data_path: Path
    test_data_path: Path
    train_images: Path
    train_labels: Path
    test_images: Path
    test_labels: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """Data Validation Configuration"""
    root_dir: Path
    status_file: Path
    required_files: list
    schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    """Data Transformation Configuration"""
    root_dir: Path
    transformed_train_dir: Path
    transformed_test_dir: Path
    augmented_data_dir: Path
    image_size: list


@dataclass(frozen=True)
class ModelTrainingConfig:
    """Model Training Configuration"""
    root_dir: Path
    model_name: str
    trained_model_path: Path
    training_data: Path
    model_config_path: Path
    params: dict


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Model Evaluation Configuration"""
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metrics_file: Path
    confusion_matrix_path: Path
    predictions_dir: Path
