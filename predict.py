"""
Command-line Inference Script
Easy-to-use CLI for fire and smoke detection
"""

import argparse
import sys
from pathlib import Path
from fire_smoke_detection.inference.predictor import FireSmokeDetector


def main():
    parser = argparse.ArgumentParser(
        description="Fire and Smoke Detection - Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect on single image
  python predict.py --image path/to/image.jpg
  
  # Detect on video
  python predict.py --video path/to/video.mp4
  
  # Detect from webcam
  python predict.py --webcam
  
  # Detect on folder of images
  python predict.py --folder path/to/images/
  
  # Adjust confidence threshold
  python predict.py --image test.jpg --conf 0.5
  
  # Use CPU instead of GPU
  python predict.py --image test.jpg --device cpu
        """
    )
    
    # Input sources
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--image', type=str, help='Path to image file')
    source_group.add_argument('--video', type=str, help='Path to video file')
    source_group.add_argument('--webcam', action='store_true', help='Use webcam')
    source_group.add_argument('--folder', type=str, help='Path to folder with images')
    
    # Model parameters
    parser.add_argument(
        '--model',
        type=str,
        default='artifacts/model_training/best.pt',
        help='Path to model weights (default: artifacts/model_training/best.pt)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use: "0" for GPU, "cpu" for CPU (default: 0)'
    )
    
    # Output parameters
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display results (for images only)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera index for webcam (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    print(f"Loading model from: {args.model}")
    detector = FireSmokeDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    # Print model info
    info = detector.get_model_info()
    print(f"Model loaded: {info['num_classes']} classes - {info['classes']}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Device: {args.device}")
    print()
    
    # Run inference based on source
    save = not args.no_save
    save_dir = Path(args.output) if args.output else None
    
    if args.image:
        print(f"Processing image: {args.image}")
        result = detector.predict_image(
            args.image,
            save=save,
            save_dir=save_dir,
            show=args.show
        )
        
        print(f"\nResults:")
        print(f"  Detections: {result['num_detections']}")
        for i, det in enumerate(result['detections'], 1):
            print(f"  {i}. {det['class']} (confidence: {det['confidence']:.2%})")
    
    elif args.video:
        print(f"Processing video: {args.video}")
        result = detector.predict_video(
            args.video,
            save=save,
            save_dir=save_dir,
            show=args.show
        )
        
        print(f"\nResults:")
        print(f"  Frames processed: {result['frames_processed']}")
        print(f"  Total detections: {result['total_detections']}")
        print(f"  Avg detections/frame: {result['avg_detections_per_frame']:.2f}")
    
    elif args.webcam:
        print(f"Starting webcam detection (camera {args.camera})")
        print("Press 'q' to quit")
        detector.predict_webcam(
            camera_index=args.camera,
            save=save,
            save_dir=save_dir
        )
    
    elif args.folder:
        print(f"Processing folder: {args.folder}")
        results = detector.predict_folder(
            args.folder,
            save=save,
            save_dir=save_dir
        )
        
        print(f"\nResults:")
        print(f"  Images processed: {len(results)}")
        total_detections = sum(r['num_detections'] for r in results)
        print(f"  Total detections: {total_detections}")
        print(f"  Avg detections/image: {total_detections/len(results):.2f}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
