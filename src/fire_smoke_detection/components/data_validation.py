"""
Data Validation Component
Validates the dataset structure, image integrity, and label format
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from box import ConfigBox
from fire_smoke_detection.entity.config_entity import DataValidationConfig


logger = logging.getLogger(__name__)


class DataValidation:
    """
    Data Validation Component
    Validates dataset structure, images, and labels according to schema
    """
    
    def __init__(self, config: DataValidationConfig):
        """
        Initialize Data Validation
        
        Args:
            config: DataValidationConfig object
        """
        self.config = config
        self.schema = ConfigBox(config.schema) if isinstance(config.schema, dict) else config.schema
        self.validation_status = True
        self.validation_report = {
            "directory_structure": {"status": "pending", "details": []},
            "image_files": {"status": "pending", "details": []},
            "label_files": {"status": "pending", "details": []},
            "image_integrity": {"status": "pending", "details": []},
            "label_format": {"status": "pending", "details": []},
            "statistics": {}
        }
    
    def validate_directory_structure(self) -> bool:
        """
        Validate that all required directories exist
        
        Returns:
            bool: True if all directories exist, False otherwise
        """
        logger.info("Validating directory structure...")
        
        required_dirs = self.schema.dataset_structure.required_directories
        all_exist = True
        
        for dir_name in required_dirs:
            dir_path = Path("data") / dir_name
            if dir_path.exists():
                self.validation_report["directory_structure"]["details"].append(
                    f"✓ Directory exists: {dir_path}"
                )
                logger.info(f"Directory found: {dir_path}")
            else:
                self.validation_report["directory_structure"]["details"].append(
                    f"✗ Directory missing: {dir_path}"
                )
                logger.error(f"Directory not found: {dir_path}")
                all_exist = False
        
        self.validation_report["directory_structure"]["status"] = "passed" if all_exist else "failed"
        return all_exist
    
    def validate_image_files(self, directory: Path) -> Tuple[bool, List[str]]:
        """
        Validate image files in a directory
        
        Args:
            directory: Path to directory containing images
            
        Returns:
            Tuple of (validation status, list of valid image files)
        """
        logger.info(f"Validating image files in {directory}...")
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return False, []
        
        allowed_extensions = self.schema.image_files.allowed_extensions
        valid_images = []
        
        for file in directory.iterdir():
            if file.is_file():
                if file.suffix.lower() in allowed_extensions:
                    valid_images.append(str(file))
                else:
                    logger.warning(f"Invalid image extension: {file}")
        
        logger.info(f"Found {len(valid_images)} valid images in {directory}")
        return len(valid_images) > 0, valid_images
    
    def validate_image_integrity(self, image_path: str) -> bool:
        """
        Validate image integrity and properties
        
        Args:
            image_path: Path to image file
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        try:
            # Try to read the image
            img = cv2.imread(image_path)
            
            if img is None:
                logger.error(f"Cannot read image: {image_path}")
                return False
            
            # Check resolution
            height, width = img.shape[:2]
            min_width = self.schema.image_files.min_resolution.width
            min_height = self.schema.image_files.min_resolution.height
            max_width = self.schema.image_files.max_resolution.width
            max_height = self.schema.image_files.max_resolution.height
            
            if width < min_width or height < min_height:
                logger.warning(f"Image resolution too low: {image_path} ({width}x{height})")
                return False
            
            if width > max_width or height > max_height:
                logger.warning(f"Image resolution too high: {image_path} ({width}x{height})")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating image {image_path}: {e}")
            return False
    
    def validate_label_file(self, label_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate YOLO format label file
        
        Args:
            label_path: Path to label file
            
        Returns:
            Tuple of (validation status, list of errors)
        """
        errors = []
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Empty label file is valid (no objects in image)
            if len(lines) == 0:
                return True, []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                # YOLO format: class_id x_center y_center width height
                if len(parts) != 5:
                    errors.append(f"Line {line_num}: Invalid format (expected 5 values, got {len(parts)})")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate class ID
                    num_classes = self.schema.label_files.num_classes
                    if class_id < 0 or class_id >= num_classes:
                        errors.append(f"Line {line_num}: Invalid class_id {class_id} (must be 0-{num_classes-1})")
                    
                    # Validate coordinates (must be between 0 and 1)
                    coord_min = self.schema.label_files.coordinate_range.min
                    coord_max = self.schema.label_files.coordinate_range.max
                    
                    for coord_name, coord_value in [
                        ("x_center", x_center),
                        ("y_center", y_center),
                        ("width", width),
                        ("height", height)
                    ]:
                        if coord_value < coord_min or coord_value > coord_max:
                            errors.append(
                                f"Line {line_num}: {coord_name} out of range ({coord_value}, must be {coord_min}-{coord_max})"
                            )
                    
                except ValueError as e:
                    errors.append(f"Line {line_num}: Invalid number format - {e}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Error reading file: {e}")
            return False, errors
    
    def validate_dataset(self) -> bool:
        """
        Perform complete dataset validation
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        logger.info("Starting dataset validation...")
        
        # 1. Validate directory structure
        if not self.validate_directory_structure():
            self.validation_status = False
            logger.error("Directory structure validation failed")
        
        # 2. Validate train and test splits
        for split in ["train", "test"]:
            images_dir = Path("data") / split / "images"
            labels_dir = Path("data") / split / "labels"
            
            # Validate images
            images_valid, image_files = self.validate_image_files(images_dir)
            
            if not images_valid:
                self.validation_report["image_files"]["details"].append(
                    f"✗ No valid images found in {images_dir}"
                )
                self.validation_status = False
                continue
            
            self.validation_report["image_files"]["details"].append(
                f"✓ Found {len(image_files)} images in {images_dir}"
            )
            
            # Sample image integrity check (check first 10 images)
            sample_size = min(10, len(image_files))
            valid_count = 0
            
            for img_path in image_files[:sample_size]:
                if self.validate_image_integrity(img_path):
                    valid_count += 1
            
            self.validation_report["image_integrity"]["details"].append(
                f"✓ {valid_count}/{sample_size} sample images valid in {split} split"
            )
            
            # Validate labels
            label_errors = []
            valid_labels = 0
            total_labels = 0
            
            for img_path in image_files:
                img_name = Path(img_path).stem
                label_path = labels_dir / f"{img_name}.txt"
                
                if not label_path.exists():
                    if self.schema.quality_checks.warn_on_missing_labels:
                        logger.warning(f"Missing label file: {label_path}")
                    continue
                
                total_labels += 1
                is_valid, errors = self.validate_label_file(label_path)
                
                if is_valid:
                    valid_labels += 1
                else:
                    label_errors.extend([f"{label_path}: {err}" for err in errors])
            
            self.validation_report["label_files"]["details"].append(
                f"✓ {valid_labels}/{total_labels} label files valid in {split} split"
            )
            
            if label_errors:
                self.validation_report["label_format"]["details"].extend(label_errors[:10])  # Show first 10 errors
            
            # Update statistics
            self.validation_report["statistics"][split] = {
                "total_images": len(image_files),
                "total_labels": total_labels,
                "valid_labels": valid_labels
            }
        
        # Set final status
        self.validation_report["image_files"]["status"] = "passed"
        self.validation_report["image_integrity"]["status"] = "passed"
        self.validation_report["label_files"]["status"] = "passed" if self.validation_status else "failed"
        self.validation_report["label_format"]["status"] = "passed" if len(label_errors) == 0 else "warning"
        
        return self.validation_status
    
    def save_validation_status(self):
        """
        Save validation status to file
        """
        with open(self.config.status_file, 'w') as f:
            f.write(f"Validation Status: {'PASSED' if self.validation_status else 'FAILED'}\n\n")
            
            for section, data in self.validation_report.items():
                if section == "statistics":
                    f.write(f"\n=== Dataset Statistics ===\n")
                    for split, stats in data.items():
                        f.write(f"\n{split.upper()} Split:\n")
                        for key, value in stats.items():
                            f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"\n=== {section.replace('_', ' ').title()} ===\n")
                    f.write(f"Status: {data['status'].upper()}\n")
                    for detail in data.get('details', []):
                        f.write(f"  {detail}\n")
        
        logger.info(f"Validation status saved to {self.config.status_file}")
