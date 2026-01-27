"""
Main Pipeline Execution Script
"""

import logging
from fire_smoke_detection.pipeline.stage_01_data_validation import DataValidationTrainingPipeline
from fire_smoke_detection.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from fire_smoke_detection.pipeline.stage_03_model_training import ModelTrainingPipeline
from fire_smoke_detection.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
)

logger = logging.getLogger(__name__)


# Stage 1: Data Validation
STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    pipeline = DataValidationTrainingPipeline()
    pipeline.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# Stage 2: Data Transformation
STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    pipeline = DataTransformationTrainingPipeline()
    pipeline.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# Stage 3: Model Training
STAGE_NAME = "Model Training Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    pipeline = ModelTrainingPipeline()
    pipeline.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# Stage 4: Model Evaluation
STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    pipeline = ModelEvaluationPipeline()
    pipeline.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
