"""
Model Evaluation Pipeline Stage
"""

import logging
from fire_smoke_detection.config.configuration import ConfigurationManager
from fire_smoke_detection.components.model_evaluation import ModelEvaluation


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
)

logger = logging.getLogger(__name__)


class ModelEvaluationPipeline:
    """
    Model Evaluation Pipeline
    """
    
    def __init__(self):
        pass
    
    def main(self):
        """
        Execute model evaluation pipeline
        """
        try:
            logger.info(">>>>>> Model Evaluation Stage Started <<<<<<")
            
            # Get configuration
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            
            # Initialize evaluation
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            
            # Evaluate the model
            metrics = model_evaluation.evaluate()
            
            # Save metrics
            model_evaluation.save_metrics(metrics)
            
            # Create visualizations
            model_evaluation.create_confusion_matrix_plot()
            model_evaluation.create_metrics_visualization(metrics)
            
            # Generate sample predictions
            model_evaluation.generate_predictions(num_samples=10)
            
            # Create evaluation report
            model_evaluation.create_evaluation_report(metrics)
            
            logger.info(">>>>>> Model Evaluation Stage Completed Successfully <<<<<<")
            logger.info(f"mAP@0.5: {metrics['mAP50']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation pipeline: {e}")
            raise e


if __name__ == '__main__':
    try:
        pipeline = ModelEvaluationPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
