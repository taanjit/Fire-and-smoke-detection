"""
Standalone Model Training Script
Run this to train the model without running the entire pipeline
"""

import logging
from fire_smoke_detection.pipeline.stage_03_model_training import ModelTrainingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    try:
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 60)
        
        pipeline = ModelTrainingPipeline()
        results = pipeline.main()
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Best model: {results['best_model_path']}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.exception(e)
        raise e
