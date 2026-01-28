"""
Configuration Manager
Handles reading and managing all configuration files
"""

from pathlib import Path
from fire_smoke_detection.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from fire_smoke_detection.utils.common import read_yaml, create_directories
from fire_smoke_detection.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig
)


class ConfigurationManager:
    """
    Configuration Manager to handle all configurations
    """
    
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        schema_filepath: Path = SCHEMA_FILE_PATH
    ):
        """
        Initialize Configuration Manager
        
        Args:
            config_filepath: Path to config.yaml
            params_filepath: Path to params.yaml
            schema_filepath: Path to schema.yaml
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        # Create artifacts root directory
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get Data Ingestion Configuration
        
        Returns:
            DataIngestionConfig: Data ingestion configuration object
        """
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_dir=Path(config.source_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            train_images=Path(config.train_images),
            train_labels=Path(config.train_labels),
            test_images=Path(config.test_images),
            test_labels=Path(config.test_labels)
        )
        
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Get Data Validation Configuration
        
        Returns:
            DataValidationConfig: Data validation configuration object
        """
        config = self.config.data_validation
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            status_file=Path(config.status_file),
            required_files=config.required_files,
            schema=dict(self.schema)
        )
        
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Get Data Transformation Configuration
        
        Returns:
            DataTransformationConfig: Data transformation configuration object
        """
        config = self.config.data_transformation
        
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            transformed_train_dir=Path(config.transformed_train_dir),
            transformed_test_dir=Path(config.transformed_test_dir),
            augmented_data_dir=Path(config.augmented_data_dir),
            image_size=config.image_size,
            schema=dict(self.schema)
        )
        
        return data_transformation_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        """
        Get Model Training Configuration
        
        Returns:
            ModelTrainingConfig: Model training configuration object
        """
        config = self.config.model_training
        
        create_directories([config.root_dir])
        
        model_training_config = ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            model_name=config.model_name,
            trained_model_path=Path(config.trained_model_path),
            training_data=Path(config.training_data),
            model_config_path=Path(config.model_config_path),
            params=dict(self.params)
        )
        
        return model_training_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Get Model Evaluation Configuration
        
        Returns:
            ModelEvaluationConfig: Model evaluation configuration object
        """
        config = self.config.model_evaluation
        
        create_directories([config.root_dir])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            metrics_file=Path(config.metrics_file),
            confusion_matrix_path=Path(config.confusion_matrix_path),
            predictions_dir=Path(config.predictions_dir)
        )
        
        return model_evaluation_config
