"""
Model Training Pipeline Stage
"""

import logging
from fire_smoke_detection.config.configuration import ConfigurationManager
from fire_smoke_detection.components.model_training import ModelTraining


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
)

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    Model Training Pipeline
    """
    
    def __init__(self):
        pass
    
    def main(self):
        """
        Execute model training pipeline
        """
        try:
            logger.info(">>>>>> Model Training Stage Started <<<<<<")
            
            # Get configuration
            config = ConfigurationManager()
            model_training_config = config.get_model_training_config()
            
            # Initialize training
            model_training = ModelTraining(config=model_training_config)
            
            # Save training configuration
            model_training.save_training_config()
            
            # Train the model
            results = model_training.train()
            
            logger.info(">>>>>> Model Training Stage Completed Successfully <<<<<<")
            logger.info(f"Best model saved at: {results['best_model_path']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise e


if __name__ == '__main__':
    try:
        pipeline = ModelTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
