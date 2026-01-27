"""
Run Complete Training Pipeline
Convenient script to run the orchestrated training pipeline
"""

import argparse
import logging
from fire_smoke_detection.pipeline.training_pipeline import TrainingPipelineOrchestrator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fire and Smoke Detection - Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py

  # Skip validation and transformation
  python run_pipeline.py --skip validation transformation

  # Start from training stage
  python run_pipeline.py --start-from training

  # Skip validation, start from transformation
  python run_pipeline.py --skip validation --start-from transformation
        """
    )
    
    parser.add_argument(
        '--skip',
        nargs='+',
        choices=['validation', 'transformation', 'training', 'evaluation'],
        help='Stages to skip'
    )
    
    parser.add_argument(
        '--start-from',
        choices=['validation', 'transformation', 'training', 'evaluation'],
        help='Stage to start from (skips all previous stages)'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Do not save pipeline execution report'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("FIRE AND SMOKE DETECTION - TRAINING PIPELINE")
        logger.info("=" * 80)
        
        # Create orchestrator
        orchestrator = TrainingPipelineOrchestrator(skip_stages=args.skip)
        
        # Run pipeline
        results = orchestrator.run_pipeline(start_from=args.start_from)
        
        # Save report
        if not args.no_report:
            orchestrator.save_pipeline_report()
        
        logger.info("Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
