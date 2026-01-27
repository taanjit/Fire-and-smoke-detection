"""
Common utility functions for the Fire and Smoke Detection Project
"""

import os
import yaml
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Union
from box import ConfigBox
from ensure import ensure_annotations
import cv2
import numpy as np


logger = logging.getLogger(__name__)


# ============================================================================
# YAML and JSON Operations
# ============================================================================

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read YAML file and return ConfigBox object
    
    Args:
        path_to_yaml (Path): Path to YAML file
        
    Returns:
        ConfigBox: ConfigBox object with YAML contents
        
    Raises:
        ValueError: If YAML file is empty
        Exception: If there's an error reading the file
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError(f"YAML file is empty: {path_to_yaml}")
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error reading YAML file {path_to_yaml}: {e}")
        raise e


@ensure_annotations
def write_yaml(path: Path, data: Dict):
    """
    Write data to YAML file
    
    Args:
        path (Path): Path to YAML file
        data (Dict): Data to write
    """
    try:
        with open(path, 'w') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False, indent=2)
        logger.info(f"YAML file saved: {path}")
    except Exception as e:
        logger.error(f"Error writing YAML file {path}: {e}")
        raise e


@ensure_annotations
def save_json(path: Path, data: Dict):
    """
    Save data to JSON file
    
    Args:
        path (Path): Path to JSON file
        data (Dict): Data to save
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load JSON file
    
    Args:
        path (Path): Path to JSON file
        
    Returns:
        ConfigBox: ConfigBox object with JSON contents
    """
    with open(path, 'r') as f:
        content = json.load(f)
    logger.info(f"JSON file loaded: {path}")
    return ConfigBox(content)


# ============================================================================
# Directory and File Operations
# ============================================================================

@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Create list of directories
    
    Args:
        path_to_directories (list): List of paths to create
        verbose (bool): Whether to log directory creation
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get size of file in KB
    
    Args:
        path (Path): Path to file
        
    Returns:
        str: Size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def get_size_mb(path: Path) -> str:
    """
    Get size of file in MB
    
    Args:
        path (Path): Path to file
        
    Returns:
        str: Size in MB
    """
    size_in_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
    return f"~ {size_in_mb} MB"


@ensure_annotations
def clean_directory(directory: Path, extensions: List[str] = None):
    """
    Clean directory by removing files with specific extensions
    
    Args:
        directory (Path): Directory to clean
        extensions (List[str]): List of extensions to remove (e.g., ['.tmp', '.log'])
                               If None, removes all files
    """
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return
    
    count = 0
    for file in directory.iterdir():
        if file.is_file():
            if extensions is None or file.suffix in extensions:
                file.unlink()
                count += 1
    
    logger.info(f"Cleaned {count} files from {directory}")


@ensure_annotations
def copy_file(source: Path, destination: Path):
    """
    Copy file from source to destination
    
    Args:
        source (Path): Source file path
        destination (Path): Destination file path
    """
    import shutil
    shutil.copy2(source, destination)
    logger.info(f"Copied {source} to {destination}")


# ============================================================================
# Image Processing Utilities
# ============================================================================

@ensure_annotations
def read_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Read image from file
    
    Args:
        image_path: Path to image file
        
    Returns:
        np.ndarray: Image array
        
    Raises:
        ValueError: If image cannot be read
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return img


@ensure_annotations
def save_image(image: np.ndarray, save_path: Union[str, Path]):
    """
    Save image to file
    
    Args:
        image: Image array
        save_path: Path to save image
    """
    cv2.imwrite(str(save_path), image)
    logger.info(f"Image saved: {save_path}")


@ensure_annotations
def get_image_info(image_path: Union[str, Path]) -> Dict:
    """
    Get image information
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dict: Image information (width, height, channels, size)
    """
    img = read_image(image_path)
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'size': get_size(Path(image_path)),
        'shape': img.shape
    }


@ensure_annotations
def resize_image_aspect_ratio(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize image maintaining aspect ratio
    
    Args:
        image: Input image
        target_size: Target size for the longer side
        
    Returns:
        np.ndarray: Resized image
    """
    h, w = image.shape[:2]
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


# ============================================================================
# System and Performance Utilities
# ============================================================================

@ensure_annotations
def get_timestamp() -> str:
    """
    Get current timestamp as string
    
    Returns:
        str: Timestamp in format YYYYMMDD_HHMMSS
    """
    return time.strftime("%Y%m%d_%H%M%S")


@ensure_annotations
def calculate_file_hash(file_path: Path, algorithm: str = 'md5') -> str:
    """
    Calculate hash of a file
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256', etc.)
        
    Returns:
        str: File hash
    """
    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


@ensure_annotations
def format_time(seconds: Union[int, float]) -> str:
    """
    Format seconds to human-readable time
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


@ensure_annotations
def count_files(directory: Path, extension: str = None) -> int:
    """
    Count files in directory
    
    Args:
        directory: Directory path
        extension: File extension to filter (e.g., '.jpg')
        
    Returns:
        int: Number of files
    """
    if not directory.exists():
        return 0
    
    if extension:
        return len(list(directory.glob(f"*{extension}")))
    else:
        return len([f for f in directory.iterdir() if f.is_file()])


@ensure_annotations
def ensure_path_exists(path: Path, is_file: bool = False):
    """
    Ensure path exists, create if necessary
    
    Args:
        path: Path to check/create
        is_file: If True, creates parent directory; if False, creates directory
    """
    if is_file:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data Validation Utilities
# ============================================================================

@ensure_annotations
def validate_file_exists(file_path: Path, file_type: str = "File") -> bool:
    """
    Validate that file exists
    
    Args:
        file_path: Path to file
        file_type: Type of file for logging
        
    Returns:
        bool: True if exists, False otherwise
    """
    if file_path.exists():
        logger.info(f"{file_type} found: {file_path}")
        return True
    else:
        logger.error(f"{file_type} not found: {file_path}")
        return False


@ensure_annotations
def validate_directory_structure(base_dir: Path, required_dirs: List[str]) -> bool:
    """
    Validate directory structure
    
    Args:
        base_dir: Base directory
        required_dirs: List of required subdirectories
        
    Returns:
        bool: True if all directories exist
    """
    all_exist = True
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            logger.error(f"Required directory missing: {dir_path}")
            all_exist = False
    return all_exist

