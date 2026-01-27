"""
Data Transformation Pipeline Stage
"""

import logging
from fire_smoke_detection.config.configuration import ConfigurationManager
from fire_smoke_detection.components.data_transformation import DataTransformation


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
)

logger = logging.getLogger(__name__)


class DataTransformationTrainingPipeline:
    """
    Data Transformation Training Pipeline
    """
    
    def __init__(self):
        pass
    
    def main(self):
        """
        Execute data transformation pipeline
        """
        try:
            logger.info(">>>>>> Data Transformation Stage Started <<<<<<")
            
            # Get configuration
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            
            # Initialize and run transformation
            data_transformation = DataTransformation(config=data_transformation_config)
            results = data_transformation.transform_dataset()
            
            # Save transformation report
            data_transformation.save_transformation_report(results)
            
            logger.info(">>>>>> Data Transformation Stage Completed Successfully <<<<<<")
            return results
            
        except Exception as e:
            logger.error(f"Error in data transformation pipeline: {e}")
            raise e


if __name__ == '__main__':
    try:
        pipeline = DataTransformationTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
