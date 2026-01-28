"""
Fire and Smoke Detection - Inference Pipeline
Real-time prediction on images, videos, and webcam streams
"""

import logging
from pathlib import Path
from typing import Union, List, Optional, Dict
import cv2
import numpy as np
import base64
import torch
from ultralytics import YOLO
from fire_smoke_detection.utils.common import ensure_path_exists, get_timestamp


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
)

logger = logging.getLogger(__name__)


class FireSmokeDetector:
    """
    Fire and Smoke Detection Inference Pipeline
    Provides easy-to-use interface for predictions
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = "artifacts/model_training/best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "0"  # "0" for GPU, "cpu" for CPU
    ):
        """
        Initialize Fire and Smoke Detector
        
        Args:
            model_path: Path to trained model weights
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ("0" for GPU, "cpu" for CPU)
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Device detection
        if device == "0" and not torch.cuda.is_available():
            logger.warning("CUDA 'device=0' requested but not available. Falling back to 'cpu'.")
            self.device = "cpu"
        else:
            self.device = device
        
        # Load model
        logger.info(f"Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        logger.info("Model loaded successfully")
        
        # Class names
        self.class_names = self.model.names
        logger.info(f"Classes: {self.class_names}")
    
    def predict_image(
        self,
        image_path: Union[str, Path],
        save: bool = True,
        save_dir: Optional[Path] = None,
        show: bool = False,
        return_base64: bool = False
    ) -> Dict:
        """
        Predict on a single image
        
        Args:
            image_path: Path to image file
            save: Whether to save result with bounding boxes
            save_dir: Directory to save results (default: runs/detect/predict)
            show: Whether to display result
            
        Returns:
            Dictionary with detection results
        """
        logger.info(f"Processing image: {image_path}")
        
        # Run prediction
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=save,
            project=str(save_dir) if save_dir else None,
            show=show
        )
        
        # Extract detections
        detections = self._extract_detections(results[0])
        
        response = {
            'image_path': str(image_path),
            'detections': detections,
            'num_detections': len(detections)
        }
        
        if return_base64:
            # Plot results on image
            plotted_img = results[0].plot()
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', plotted_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response['image_base64'] = img_base64
            
        logger.info(f"Found {len(detections)} detections")
        return response
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        save: bool = True,
        save_dir: Optional[Path] = None
    ) -> List[Dict]:
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
            save: Whether to save results
            save_dir: Directory to save results
            
        Returns:
            List of detection results
        """
        logger.info(f"Processing {len(image_paths)} images...")
        
        all_results = []
        for img_path in image_paths:
            result = self.predict_image(img_path, save=save, save_dir=save_dir)
            all_results.append(result)
        
        logger.info(f"Processed {len(all_results)} images")
        return all_results
    
    def predict_video(
        self,
        video_path: Union[str, Path],
        save: bool = True,
        save_dir: Optional[Path] = None,
        show: bool = False
    ) -> Dict:
        """
        Predict on video file
        
        Args:
            video_path: Path to video file
            save: Whether to save result video
            save_dir: Directory to save results
            show: Whether to display results
            
        Returns:
            Dictionary with video processing results
        """
        logger.info(f"Processing video: {video_path}")
        
        # Run prediction on video
        results = self.model.predict(
            source=str(video_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=save,
            project=str(save_dir) if save_dir else None,
            show=show,
            stream=True  # Process frame by frame
        )
        
        # Count detections across frames
        total_detections = 0
        frame_count = 0
        
        for result in results:
            detections = self._extract_detections(result)
            total_detections += len(detections)
            frame_count += 1
        
        logger.info(f"Processed {frame_count} frames with {total_detections} total detections")
        
        return {
            'video_path': str(video_path),
            'frames_processed': frame_count,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / frame_count if frame_count > 0 else 0
        }
    
    def predict_webcam(
        self,
        camera_index: int = 0,
        save: bool = False,
        save_dir: Optional[Path] = None
    ):
        """
        Real-time prediction from webcam
        
        Args:
            camera_index: Camera device index (0 for default webcam)
            save: Whether to save video
            save_dir: Directory to save results
        """
        logger.info(f"Starting webcam detection (camera {camera_index})")
        logger.info("Press 'q' to quit")
        
        # Run prediction on webcam
        results = self.model.predict(
            source=camera_index,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=save,
            project=str(save_dir) if save_dir else None,
            show=True,  # Always show for webcam
            stream=True
        )
        
        # Process stream
        for result in results:
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        logger.info("Webcam detection stopped")
    
    def predict_folder(
        self,
        folder_path: Union[str, Path],
        save: bool = True,
        save_dir: Optional[Path] = None
    ) -> List[Dict]:
        """
        Predict on all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            save: Whether to save results
            save_dir: Directory to save results
            
        Returns:
            List of detection results
        """
        folder_path = Path(folder_path)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f'*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_files)} images in {folder_path}")
        
        return self.predict_batch(image_files, save=save, save_dir=save_dir)
    
    def _extract_detections(self, result) -> List[Dict]:
        """
        Extract detection information from YOLO result
        
        Args:
            result: YOLO result object
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                
                # Get class and confidence
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                
                detection = {
                    'class': self.class_names[cls],
                    'class_id': cls,
                    'confidence': conf,
                    'bbox': {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3])
                    }
                }
                
                detections.append(detection)
        
        return detections
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': str(self.model_path),
            'classes': self.class_names,
            'num_classes': len(self.class_names),
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'device': self.device
        }


def main():
    """
    Example usage of FireSmokeDetector
    """
    # Initialize detector
    detector = FireSmokeDetector(
        model_path="artifacts/model_training/best.pt",
        conf_threshold=0.25
    )
    
    # Print model info
    info = detector.get_model_info()
    logger.info(f"Model Info: {info}")
    
    # Example: Predict on single image
    # result = detector.predict_image("path/to/image.jpg", save=True)
    # logger.info(f"Detections: {result['detections']}")
    
    logger.info("Detector initialized and ready for inference")


if __name__ == '__main__':
    main()
