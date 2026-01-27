"""
Data Validation Pipeline
"""

import logging
from fire_smoke_detection.config.configuration import ConfigurationManager
from fire_smoke_detection.components.data_validation import DataValidation


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
)

logger = logging.getLogger(__name__)


class DataValidationTrainingPipeline:
    """
    Data Validation Training Pipeline
    """
    
    def __init__(self):
        pass
    
    def main(self):
        """
        Execute data validation pipeline
        """
        try:
            logger.info(">>>>>> Data Validation Stage Started <<<<<<")
            
            # Get configuration
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            
            # Initialize and run validation
            data_validation = DataValidation(config=data_validation_config)
            validation_passed = data_validation.validate_dataset()
            
            # Save validation status
            data_validation.save_validation_status()
            
            if validation_passed:
                logger.info(">>>>>> Data Validation Stage Completed Successfully <<<<<<")
            else:
                logger.error(">>>>>> Data Validation Stage Failed <<<<<<")
                logger.error("Please check the validation report for details")
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"Error in data validation pipeline: {e}")
            raise e


if __name__ == '__main__':
    try:
        pipeline = DataValidationTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
