"""
Standalone Model Evaluation Script
Run this to evaluate a trained model
"""

import logging
from fire_smoke_detection.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    try:
        logger.info("=" * 60)
        logger.info("STARTING MODEL EVALUATION")
        logger.info("=" * 60)
        
        pipeline = ModelEvaluationPipeline()
        metrics = pipeline.main()
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info(f"mAP@0.5: {metrics['mAP50']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.exception(e)
        raise e
