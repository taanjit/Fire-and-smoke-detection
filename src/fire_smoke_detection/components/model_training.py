"""
Model Training Component
Implements YOLOv8 training logic for fire and smoke detection
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional
import yaml
from ultralytics import YOLO
from fire_smoke_detection.entity.config_entity import ModelTrainingConfig


logger = logging.getLogger(__name__)


class ModelTraining:
    """
    Model Training Component
    Trains YOLOv8 model for fire and smoke detection
    """
    
    def __init__(self, config: ModelTrainingConfig):
        """
        Initialize Model Training
        
        Args:
            config: ModelTrainingConfig object
        """
        self.config = config
        self.params = config.params
        self.model = None
        
    def get_base_model(self) -> YOLO:
        """
        Get base YOLOv8 model
        
        Returns:
            YOLO model instance
        """
        model_config = self.params.get('MODEL', {})
        architecture = model_config.get('architecture', 'yolov8')
        variant = model_config.get('variant', 'n')
        pretrained = model_config.get('pretrained', True)
        
        # Construct model name (e.g., yolov8n.pt)
        model_name = f"{architecture}{variant}.pt"
        
        logger.info(f"Loading base model: {model_name}")
        
        if pretrained:
            # Load pretrained model
            model = YOLO(model_name)
            logger.info(f"Loaded pretrained {model_name}")
        else:
            # Load model architecture only (no pretrained weights)
            model = YOLO(f"{architecture}{variant}.yaml")
            logger.info(f"Loaded {model_name} architecture without pretrained weights")
        
        return model
    
    def prepare_training_config(self) -> Dict:
        """
        Prepare training configuration from params
        
        Returns:
            Dictionary of training arguments
        """
        training_params = self.params.get('TRAINING', {})
        augmentation_params = self.params.get('AUGMENTATION', {})
        image_params = self.params.get('IMAGE', {})
        
        # Prepare training arguments for YOLO
        train_args = {
            # Data
            'data': str(Path('artifacts/data_transformation/dataset.yaml').absolute()),
            
            # Training hyperparameters
            'epochs': training_params.get('epochs', 100),
            'batch': training_params.get('batch_size', 16),
            'imgsz': image_params.get('input_size', 640),
            
            # Optimizer
            'optimizer': training_params.get('optimizer', 'SGD'),
            'lr0': training_params.get('learning_rate', 0.01),
            'lrf': training_params.get('lr_decay_factor', 0.1),
            'momentum': training_params.get('momentum', 0.937),
            'weight_decay': training_params.get('weight_decay', 0.0005),
            
            # Warmup
            'warmup_epochs': training_params.get('warmup_epochs', 3),
            'warmup_momentum': training_params.get('warmup_momentum', 0.8),
            'warmup_bias_lr': training_params.get('warmup_bias_lr', 0.1),
            
            # Augmentation
            'hsv_h': augmentation_params.get('hsv_h', 0.015),
            'hsv_s': augmentation_params.get('hsv_s', 0.7),
            'hsv_v': augmentation_params.get('hsv_v', 0.4),
            'degrees': augmentation_params.get('degrees', 0.0),
            'translate': augmentation_params.get('translate', 0.1),
            'scale': augmentation_params.get('scale', 0.5),
            'shear': augmentation_params.get('shear', 0.0),
            'perspective': augmentation_params.get('perspective', 0.0),
            'flipud': augmentation_params.get('flipud', 0.0),
            'fliplr': augmentation_params.get('fliplr', 0.5),
            'mosaic': augmentation_params.get('mosaic', 1.0),
            'mixup': augmentation_params.get('mixup', 0.0),
            'copy_paste': augmentation_params.get('copy_paste', 0.0),
            
            # Device and workers
            'device': training_params.get('device', 0),
            'workers': training_params.get('workers', 8),
            
            # Saving
            'project': str(self.config.root_dir),
            'name': 'train',
            'exist_ok': True,
            'save': True,
            'save_period': self.params.get('VALIDATION', {}).get('save_period', 10),
            
            # Validation
            'val': True,
            
            # Mixed precision
            'amp': training_params.get('amp', True),
            
            # Early stopping
            'patience': training_params.get('patience', 50),
            
            # Logging
            'verbose': True,
            'plots': self.params.get('LOGGING', {}).get('save_plots', True),
        }
        
        return train_args
    
    def train(self) -> Dict:
        """
        Train the model
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training...")
        
        # Get base model
        self.model = self.get_base_model()
        
        # Prepare training configuration
        train_args = self.prepare_training_config()
        
        logger.info("Training configuration:")
        for key, value in train_args.items():
            logger.info(f"  {key}: {value}")
        
        # Train the model
        logger.info("Training started...")
        results = self.model.train(**train_args)
        
        logger.info("Training completed!")
        
        # Get best model path
        best_model_path = Path(self.config.root_dir) / 'train' / 'weights' / 'best.pt'
        
        # Copy best model to configured path
        if best_model_path.exists():
            import shutil
            shutil.copy(best_model_path, self.config.trained_model_path)
            logger.info(f"Best model saved to: {self.config.trained_model_path}")
        
        return {
            'best_model_path': str(self.config.trained_model_path),
            'training_dir': str(Path(self.config.root_dir) / 'train'),
            'results': results
        }
    
    def save_training_config(self):
        """
        Save training configuration for reference
        """
        config_data = {
            'model': {
                'architecture': self.params.get('MODEL', {}).get('architecture', 'yolov8'),
                'variant': self.params.get('MODEL', {}).get('variant', 'n'),
                'pretrained': self.params.get('MODEL', {}).get('pretrained', True),
            },
            'training': self.params.get('TRAINING', {}),
            'augmentation': self.params.get('AUGMENTATION', {}),
            'image': self.params.get('IMAGE', {}),
        }
        
        config_path = self.config.model_config_path
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        logger.info(f"Training configuration saved to: {config_path}")
    
    def load_trained_model(self, model_path: Optional[Path] = None) -> YOLO:
        """
        Load a trained model
        
        Args:
            model_path: Path to model weights (uses config path if None)
            
        Returns:
            Loaded YOLO model
        """
        if model_path is None:
            model_path = self.config.trained_model_path
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        logger.info(f"Loading trained model from: {model_path}")
        model = YOLO(str(model_path))
        
        return model
