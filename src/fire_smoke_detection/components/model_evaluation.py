"""
Model Evaluation Component
Implements comprehensive evaluation metrics for object detection
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from fire_smoke_detection.entity.config_entity import ModelEvaluationConfig


logger = logging.getLogger(__name__)


class ModelEvaluation:
    """
    Model Evaluation Component
    Evaluates trained model with comprehensive metrics
    """
    
    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize Model Evaluation
        
        Args:
            config: ModelEvaluationConfig object
        """
        self.config = config
        self.model = None
        self.results = None
        
    def load_model(self):
        """
        Load trained model for evaluation
        """
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.config.model_path}")
        
        logger.info(f"Loading model from: {self.config.model_path}")
        self.model = YOLO(str(self.config.model_path))
        logger.info("Model loaded successfully")
    
    def evaluate(self) -> Dict:
        """
        Perform comprehensive model evaluation
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting model evaluation...")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Run validation on test data
        logger.info(f"Evaluating on test data: {self.config.test_data_path}")
        
        # YOLO validation returns metrics
        self.results = self.model.val(
            data=str(Path('artifacts/data_transformation/dataset.yaml').absolute()),
            split='test',
            save_json=True,
            save_hybrid=True,
            conf=0.25,
            iou=0.45,
            plots=True,
            verbose=True
        )
        
        # Extract metrics
        metrics = self.extract_metrics()
        
        logger.info("Evaluation completed!")
        return metrics
    
    def extract_metrics(self) -> Dict:
        """
        Extract and organize evaluation metrics
        
        Returns:
            Dictionary of metrics
        """
        if self.results is None:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        
        # Extract metrics from YOLO results
        metrics = {
            'mAP50': float(self.results.box.map50),  # mAP at IoU=0.50
            'mAP50-95': float(self.results.box.map),  # mAP at IoU=0.50:0.95
            'precision': float(self.results.box.mp),  # Mean precision
            'recall': float(self.results.box.mr),  # Mean recall
            'f1_score': self.calculate_f1(
                float(self.results.box.mp),
                float(self.results.box.mr)
            ),
        }
        
        # Per-class metrics
        if hasattr(self.results.box, 'ap_class_index'):
            class_names = ['fire', 'smoke', 'both']
            metrics['per_class'] = {}
            
            for idx, class_name in enumerate(class_names):
                if idx < len(self.results.box.ap):
                    metrics['per_class'][class_name] = {
                        'precision': float(self.results.box.p[idx]) if idx < len(self.results.box.p) else 0.0,
                        'recall': float(self.results.box.r[idx]) if idx < len(self.results.box.r) else 0.0,
                        'mAP50': float(self.results.box.ap50[idx]) if idx < len(self.results.box.ap50) else 0.0,
                        'mAP50-95': float(self.results.box.ap[idx]) if idx < len(self.results.box.ap) else 0.0,
                    }
        
        logger.info("Metrics extracted:")
        logger.info(f"  mAP50: {metrics['mAP50']:.4f}")
        logger.info(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def calculate_f1(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall
        
        Args:
            precision: Precision value
            recall: Recall value
            
        Returns:
            F1 score
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def save_metrics(self, metrics: Dict):
        """
        Save evaluation metrics to JSON file
        
        Args:
            metrics: Dictionary of metrics
        """
        metrics_file = self.config.metrics_file
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Metrics saved to: {metrics_file}")
    
    def create_confusion_matrix_plot(self):
        """
        Create and save confusion matrix visualization
        """
        logger.info("Creating confusion matrix plot...")
        
        # YOLO automatically saves confusion matrix during validation
        # We'll create an additional custom visualization
        
        try:
            # Check if YOLO generated confusion matrix
            val_dir = Path(self.config.model_path).parent.parent / 'train'
            confusion_matrix_path = val_dir / 'confusion_matrix.png'
            
            if confusion_matrix_path.exists():
                import shutil
                shutil.copy(confusion_matrix_path, self.config.confusion_matrix_path)
                logger.info(f"Confusion matrix saved to: {self.config.confusion_matrix_path}")
            else:
                logger.warning("Confusion matrix not found in validation results")
        
        except Exception as e:
            logger.error(f"Error creating confusion matrix: {e}")
    
    def create_metrics_visualization(self, metrics: Dict):
        """
        Create visualization of evaluation metrics
        
        Args:
            metrics: Dictionary of metrics
        """
        logger.info("Creating metrics visualization...")
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Overall metrics bar chart
            overall_metrics = {
                'mAP50': metrics['mAP50'],
                'mAP50-95': metrics['mAP50-95'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            }
            
            axes[0].bar(overall_metrics.keys(), overall_metrics.values(), color='steelblue')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Overall Evaluation Metrics')
            axes[0].set_ylim([0, 1])
            axes[0].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (key, value) in enumerate(overall_metrics.items()):
                axes[0].text(i, value + 0.02, f'{value:.3f}', ha='center', va='bottom')
            
            # Plot 2: Per-class mAP50
            if 'per_class' in metrics:
                class_names = list(metrics['per_class'].keys())
                map50_values = [metrics['per_class'][c]['mAP50'] for c in class_names]
                
                axes[1].bar(class_names, map50_values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
                axes[1].set_ylabel('mAP50')
                axes[1].set_title('Per-Class mAP50')
                axes[1].set_ylim([0, 1])
                axes[1].grid(axis='y', alpha=0.3)
                
                # Add value labels
                for i, value in enumerate(map50_values):
                    axes[1].text(i, value + 0.02, f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.config.root_dir / 'evaluation_metrics.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Metrics visualization saved to: {plot_path}")
        
        except Exception as e:
            logger.error(f"Error creating metrics visualization: {e}")
    
    def generate_predictions(self, num_samples: int = 10):
        """
        Generate predictions on test samples and save visualizations
        
        Args:
            num_samples: Number of samples to visualize
        """
        logger.info(f"Generating predictions on {num_samples} test samples...")
        
        try:
            # Get test images
            test_images_dir = Path('artifacts/data_transformation/test/images')
            test_images = list(test_images_dir.glob('*.jpg'))[:num_samples]
            
            # Create predictions directory
            self.config.predictions_dir.mkdir(parents=True, exist_ok=True)
            
            # Run predictions
            for img_path in test_images:
                results = self.model.predict(
                    source=str(img_path),
                    conf=0.25,
                    iou=0.45,
                    save=True,
                    project=str(self.config.predictions_dir),
                    name='',
                    exist_ok=True
                )
            
            logger.info(f"Predictions saved to: {self.config.predictions_dir}")
        
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
    
    def create_evaluation_report(self, metrics: Dict):
        """
        Create comprehensive evaluation report
        
        Args:
            metrics: Dictionary of metrics
        """
        report_path = self.config.root_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write(f"  mAP@0.5:      {metrics['mAP50']:.4f}\n")
            f.write(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}\n")
            f.write(f"  Precision:    {metrics['precision']:.4f}\n")
            f.write(f"  Recall:       {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:     {metrics['f1_score']:.4f}\n\n")
            
            if 'per_class' in metrics:
                f.write("PER-CLASS METRICS:\n")
                for class_name, class_metrics in metrics['per_class'].items():
                    f.write(f"\n  {class_name.upper()}:\n")
                    f.write(f"    Precision:    {class_metrics['precision']:.4f}\n")
                    f.write(f"    Recall:       {class_metrics['recall']:.4f}\n")
                    f.write(f"    mAP@0.5:      {class_metrics['mAP50']:.4f}\n")
                    f.write(f"    mAP@0.5:0.95: {class_metrics['mAP50-95']:.4f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Model Path: {self.config.model_path}\n")
            f.write(f"Test Data: {self.config.test_data_path}\n")
            f.write("=" * 60 + "\n")
        
        logger.info(f"Evaluation report saved to: {report_path}")
