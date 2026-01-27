"""
Data Transformation Component
Handles preprocessing, augmentation, and dataset preparation for YOLO training
"""

import os
import shutil
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm
from fire_smoke_detection.entity.config_entity import DataTransformationConfig


logger = logging.getLogger(__name__)


class DataTransformation:
    """
    Data Transformation Component
    Prepares dataset for YOLO training with preprocessing and augmentation
    """
    
    def __init__(self, config: DataTransformationConfig):
        """
        Initialize Data Transformation
        
        Args:
            config: DataTransformationConfig object
        """
        self.config = config
        self.image_size = tuple(config.image_size)
        
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio with letterboxing
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            
        Returns:
            Resized image with letterboxing
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create letterboxed image
        letterboxed = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        # Calculate padding
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        # Place resized image in center
        letterboxed[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        return letterboxed, scale, (pad_w, pad_h)
    
    def adjust_labels(self, label_path: Path, scale: float, padding: Tuple[int, int], 
                     original_size: Tuple[int, int], target_size: Tuple[int, int]) -> List[str]:
        """
        Adjust YOLO labels for resized and padded images
        
        Args:
            label_path: Path to original label file
            scale: Scaling factor applied to image
            padding: Padding (pad_w, pad_h)
            original_size: Original image size (width, height)
            target_size: Target image size (width, height)
            
        Returns:
            List of adjusted label lines
        """
        if not label_path.exists():
            return []
        
        adjusted_labels = []
        orig_w, orig_h = original_size
        target_w, target_h = target_size
        pad_w, pad_h = padding
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert from normalized to absolute coordinates
            abs_x = x_center * orig_w
            abs_y = y_center * orig_h
            abs_w = width * orig_w
            abs_h = height * orig_h
            
            # Apply scaling and padding
            new_x = abs_x * scale + pad_w
            new_y = abs_y * scale + pad_h
            new_w = abs_w * scale
            new_h = abs_h * scale
            
            # Convert back to normalized coordinates
            norm_x = new_x / target_w
            norm_y = new_y / target_h
            norm_w = new_w / target_w
            norm_h = new_h / target_h
            
            # Ensure values are within [0, 1]
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))
            
            adjusted_labels.append(f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        return adjusted_labels
    
    def transform_split(self, split_name: str, source_images: Path, source_labels: Path,
                       dest_images: Path, dest_labels: Path) -> Dict[str, int]:
        """
        Transform a dataset split (train/test)
        
        Args:
            split_name: Name of the split (train/test)
            source_images: Source images directory
            source_labels: Source labels directory
            dest_images: Destination images directory
            dest_labels: Destination labels directory
            
        Returns:
            Dictionary with transformation statistics
        """
        logger.info(f"Transforming {split_name} split...")
        
        # Create destination directories
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)
        
        stats = {
            "total_images": 0,
            "processed_images": 0,
            "skipped_images": 0,
            "total_labels": 0
        }
        
        # Get all image files
        image_files = list(source_images.glob("*.jpg")) + \
                     list(source_images.glob("*.jpeg")) + \
                     list(source_images.glob("*.png"))
        
        stats["total_images"] = len(image_files)
        
        for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
            try:
                # Read image
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Cannot read image: {img_path}")
                    stats["skipped_images"] += 1
                    continue
                
                original_h, original_w = image.shape[:2]
                
                # Resize image with letterboxing
                transformed_image, scale, padding = self.resize_image(image, self.image_size)
                
                # Save transformed image
                dest_img_path = dest_images / img_path.name
                cv2.imwrite(str(dest_img_path), transformed_image)
                
                # Process corresponding label file
                label_path = source_labels / f"{img_path.stem}.txt"
                if label_path.exists():
                    adjusted_labels = self.adjust_labels(
                        label_path, scale, padding,
                        (original_w, original_h), self.image_size
                    )
                    
                    # Save adjusted labels
                    dest_label_path = dest_labels / f"{img_path.stem}.txt"
                    with open(dest_label_path, 'w') as f:
                        f.write('\n'.join(adjusted_labels))
                    
                    stats["total_labels"] += len(adjusted_labels)
                else:
                    # Create empty label file for images without annotations
                    dest_label_path = dest_labels / f"{img_path.stem}.txt"
                    dest_label_path.touch()
                
                stats["processed_images"] += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                stats["skipped_images"] += 1
        
        logger.info(f"{split_name} split transformation complete:")
        logger.info(f"  Processed: {stats['processed_images']}/{stats['total_images']}")
        logger.info(f"  Total labels: {stats['total_labels']}")
        
        return stats
    
    def create_yolo_config(self, train_path: Path, test_path: Path, 
                          num_classes: int, class_names: List[str]) -> Path:
        """
        Create YOLO dataset configuration file
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            num_classes: Number of classes
            class_names: List of class names
            
        Returns:
            Path to created config file
        """
        config_path = self.config.root_dir / "dataset.yaml"
        
        yolo_config = {
            'path': str(self.config.root_dir.absolute()),
            'train': str(train_path.relative_to(self.config.root_dir)),
            'val': str(test_path.relative_to(self.config.root_dir)),
            'test': str(test_path.relative_to(self.config.root_dir)),
            'nc': num_classes,
            'names': class_names
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(yolo_config, f, default_flow_style=False)
        
        logger.info(f"YOLO dataset config created at: {config_path}")
        return config_path
    
    def transform_dataset(self) -> Dict[str, any]:
        """
        Perform complete dataset transformation
        
        Returns:
            Dictionary with transformation results
        """
        logger.info("Starting dataset transformation...")
        
        results = {
            "train": {},
            "test": {},
            "config_path": None
        }
        
        # Transform training data
        results["train"] = self.transform_split(
            "train",
            Path("data/train/images"),
            Path("data/train/labels"),
            self.config.transformed_train_dir / "images",
            self.config.transformed_train_dir / "labels"
        )
        
        # Transform test data
        results["test"] = self.transform_split(
            "test",
            Path("data/test/images"),
            Path("data/test/labels"),
            self.config.transformed_test_dir / "images",
            self.config.transformed_test_dir / "labels"
        )
        
        # Create YOLO dataset configuration
        # Note: Update num_classes and class_names based on your schema
        class_names = ['fire', 'smoke', 'both']  # Update this based on your classes
        num_classes = len(class_names)
        
        results["config_path"] = self.create_yolo_config(
            self.config.transformed_train_dir,
            self.config.transformed_test_dir,
            num_classes,
            class_names
        )
        
        logger.info("Dataset transformation completed successfully!")
        return results
    
    def save_transformation_report(self, results: Dict[str, any]):
        """
        Save transformation report
        
        Args:
            results: Transformation results dictionary
        """
        report_path = self.config.root_dir / "transformation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATA TRANSFORMATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Target Image Size: {self.image_size}\n\n")
            
            f.write("TRAINING DATA:\n")
            f.write(f"  Total Images: {results['train']['total_images']}\n")
            f.write(f"  Processed: {results['train']['processed_images']}\n")
            f.write(f"  Skipped: {results['train']['skipped_images']}\n")
            f.write(f"  Total Labels: {results['train']['total_labels']}\n\n")
            
            f.write("TEST DATA:\n")
            f.write(f"  Total Images: {results['test']['total_images']}\n")
            f.write(f"  Processed: {results['test']['processed_images']}\n")
            f.write(f"  Skipped: {results['test']['skipped_images']}\n")
            f.write(f"  Total Labels: {results['test']['total_labels']}\n\n")
            
            f.write(f"YOLO Config: {results['config_path']}\n")
        
        logger.info(f"Transformation report saved to: {report_path}")
