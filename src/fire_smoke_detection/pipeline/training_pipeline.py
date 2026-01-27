"""
Complete Training Pipeline Orchestrator
Orchestrates all pipeline stages with progress tracking and error handling
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from fire_smoke_detection.pipeline.stage_01_data_validation import DataValidationTrainingPipeline
from fire_smoke_detection.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from fire_smoke_detection.pipeline.stage_03_model_training import ModelTrainingPipeline
from fire_smoke_detection.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline
from fire_smoke_detection.utils.common import get_timestamp, format_time, create_directories


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
)

logger = logging.getLogger(__name__)


class TrainingPipelineOrchestrator:
    """
    Complete Training Pipeline Orchestrator
    Manages execution of all pipeline stages with tracking and error handling
    """
    
    def __init__(self, skip_stages: Optional[List[str]] = None):
        """
        Initialize Pipeline Orchestrator
        
        Args:
            skip_stages: List of stage names to skip (e.g., ['validation', 'transformation'])
        """
        self.skip_stages = skip_stages or []
        self.pipeline_start_time = None
        self.stage_times = {}
        self.results = {}
        
        # Define pipeline stages
        self.stages = {
            'validation': {
                'name': 'Data Validation',
                'pipeline': DataValidationTrainingPipeline,
                'description': 'Validate dataset structure and integrity'
            },
            'transformation': {
                'name': 'Data Transformation',
                'pipeline': DataTransformationTrainingPipeline,
                'description': 'Preprocess and transform dataset for training'
            },
            'training': {
                'name': 'Model Training',
                'pipeline': ModelTrainingPipeline,
                'description': 'Train YOLOv8 model on transformed dataset'
            },
            'evaluation': {
                'name': 'Model Evaluation',
                'pipeline': ModelEvaluationPipeline,
                'description': 'Evaluate trained model and generate metrics'
            }
        }
    
    def print_banner(self, text: str, char: str = "="):
        """Print formatted banner"""
        width = 80
        logger.info(char * width)
        logger.info(f"{text:^{width}}")
        logger.info(char * width)
    
    def print_stage_header(self, stage_name: str, description: str):
        """Print stage header"""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STAGE: {stage_name}")
        logger.info(f"DESCRIPTION: {description}")
        logger.info("=" * 80)
    
    def run_stage(self, stage_key: str) -> bool:
        """
        Run a single pipeline stage
        
        Args:
            stage_key: Stage identifier (e.g., 'validation', 'training')
            
        Returns:
            bool: True if stage completed successfully
        """
        if stage_key in self.skip_stages:
            logger.info(f"Skipping {stage_key} stage (as requested)")
            return True
        
        stage_info = self.stages[stage_key]
        stage_name = stage_info['name']
        
        try:
            # Print stage header
            self.print_stage_header(stage_name, stage_info['description'])
            
            # Start timing
            start_time = time.time()
            
            # Run stage
            logger.info(f">>>>>> {stage_name} Stage Started <<<<<<")
            pipeline = stage_info['pipeline']()
            result = pipeline.main()
            
            # Calculate duration
            duration = time.time() - start_time
            self.stage_times[stage_key] = duration
            self.results[stage_key] = result
            
            # Log completion
            logger.info(f">>>>>> {stage_name} Stage Completed <<<<<<")
            logger.info(f"Duration: {format_time(duration)}")
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in {stage_name} stage: {e}")
            logger.exception(e)
            return False
    
    def run_pipeline(self, start_from: Optional[str] = None) -> Dict:
        """
        Run complete training pipeline
        
        Args:
            start_from: Stage to start from (e.g., 'training' to skip validation and transformation)
            
        Returns:
            Dict: Pipeline execution results
        """
        self.pipeline_start_time = time.time()
        
        # Print pipeline header
        self.print_banner("FIRE AND SMOKE DETECTION - TRAINING PIPELINE")
        logger.info(f"Pipeline Start Time: {get_timestamp()}")
        logger.info(f"Skip Stages: {self.skip_stages if self.skip_stages else 'None'}")
        logger.info(f"Start From: {start_from if start_from else 'Beginning'}")
        logger.info("")
        
        # Determine which stages to run
        stage_order = ['validation', 'transformation', 'training', 'evaluation']
        
        if start_from:
            try:
                start_index = stage_order.index(start_from)
                stage_order = stage_order[start_index:]
                logger.info(f"Starting from {start_from} stage")
            except ValueError:
                logger.warning(f"Invalid start_from stage: {start_from}. Starting from beginning.")
        
        # Run stages
        for stage_key in stage_order:
            success = self.run_stage(stage_key)
            
            if not success:
                logger.error(f"Pipeline failed at {stage_key} stage")
                self.print_pipeline_summary(failed_stage=stage_key)
                raise RuntimeError(f"Pipeline failed at {stage_key} stage")
        
        # Print summary
        self.print_pipeline_summary()
        
        return {
            'success': True,
            'stage_times': self.stage_times,
            'results': self.results,
            'total_time': time.time() - self.pipeline_start_time
        }
    
    def print_pipeline_summary(self, failed_stage: Optional[str] = None):
        """
        Print pipeline execution summary
        
        Args:
            failed_stage: Stage that failed (if any)
        """
        total_time = time.time() - self.pipeline_start_time
        
        logger.info("")
        self.print_banner("PIPELINE EXECUTION SUMMARY")
        
        if failed_stage:
            logger.error(f"Status: FAILED at {failed_stage} stage")
        else:
            logger.info("Status: COMPLETED SUCCESSFULLY")
        
        logger.info("")
        logger.info("Stage Execution Times:")
        logger.info("-" * 80)
        
        for stage_key, duration in self.stage_times.items():
            stage_name = self.stages[stage_key]['name']
            percentage = (duration / total_time) * 100
            logger.info(f"  {stage_name:.<40} {format_time(duration):>10} ({percentage:>5.1f}%)")
        
        logger.info("-" * 80)
        logger.info(f"  {'Total Pipeline Time':.<40} {format_time(total_time):>10} (100.0%)")
        logger.info("")
        
        if not failed_stage and 'evaluation' in self.results:
            metrics = self.results['evaluation']
            logger.info("Final Model Metrics:")
            logger.info("-" * 80)
            logger.info(f"  mAP@0.5:      {metrics.get('mAP50', 0):.4f}")
            logger.info(f"  mAP@0.5:0.95: {metrics.get('mAP50-95', 0):.4f}")
            logger.info(f"  Precision:    {metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall:       {metrics.get('recall', 0):.4f}")
            logger.info(f"  F1-Score:     {metrics.get('f1_score', 0):.4f}")
            logger.info("")
        
        self.print_banner("END OF PIPELINE")
    
    def save_pipeline_report(self, output_path: Optional[Path] = None):
        """
        Save pipeline execution report
        
        Args:
            output_path: Path to save report (default: artifacts/pipeline_report.txt)
        """
        if output_path is None:
            output_path = Path("artifacts/pipeline_report.txt")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_time = time.time() - self.pipeline_start_time
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FIRE AND SMOKE DETECTION - TRAINING PIPELINE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Execution Time: {get_timestamp()}\n")
            f.write(f"Total Duration: {format_time(total_time)}\n\n")
            
            f.write("STAGE EXECUTION TIMES:\n")
            f.write("-" * 80 + "\n")
            
            for stage_key, duration in self.stage_times.items():
                stage_name = self.stages[stage_key]['name']
                percentage = (duration / total_time) * 100
                f.write(f"  {stage_name:.<40} {format_time(duration):>10} ({percentage:>5.1f}%)\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"  {'Total':.<40} {format_time(total_time):>10} (100.0%)\n\n")
            
            if 'evaluation' in self.results:
                metrics = self.results['evaluation']
                f.write("FINAL MODEL METRICS:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  mAP@0.5:      {metrics.get('mAP50', 0):.4f}\n")
                f.write(f"  mAP@0.5:0.95: {metrics.get('mAP50-95', 0):.4f}\n")
                f.write(f"  Precision:    {metrics.get('precision', 0):.4f}\n")
                f.write(f"  Recall:       {metrics.get('recall', 0):.4f}\n")
                f.write(f"  F1-Score:     {metrics.get('f1_score', 0):.4f}\n\n")
            
            f.write("=" * 80 + "\n")
        
        logger.info(f"Pipeline report saved to: {output_path}")


def main():
    """
    Main function to run the complete training pipeline
    """
    try:
        # Create orchestrator
        orchestrator = TrainingPipelineOrchestrator()
        
        # Run pipeline
        results = orchestrator.run_pipeline()
        
        # Save report
        orchestrator.save_pipeline_report()
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise e


if __name__ == '__main__':
    main()
