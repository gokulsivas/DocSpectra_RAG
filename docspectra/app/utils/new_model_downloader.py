#!/usr/bin/env pyth
"""
Download all marker models to a specified directory for AWS deployment.
Run this script to pre-download models before deploying to AWS.
"""

import os
import sys
import logging
import time
from pathlib import Path
from marker.models import create_model_dict
from marker.settings import Settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_models_to_directory(target_dir: str = "/opt/marker_models"):
    """
    Download all marker models to a specific directory.
    
    Args:
        target_dir (str): Directory where models will be stored
    """
    start_time = time.time()
    logger.info(f"=== MARKER MODEL DOWNLOAD STARTED ===")
    logger.info(f"Target directory: {target_dir}")
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Created directory: {target_dir}")
    
    # Set environment variables for model cache location
    original_cache = os.environ.get('TRANSFORMERS_CACHE')
    original_torch_home = os.environ.get('TORCH_HOME')
    original_hf_home = os.environ.get('HF_HOME')
    
    try:
        # Override cache directories to our target
        os.environ['TRANSFORMERS_CACHE'] = target_dir
        os.environ['TORCH_HOME'] = target_dir
        os.environ['HF_HOME'] = target_dir
        
        # Also set marker-specific cache paths
        cache_dir = Path(target_dir) / "models"
        cache_dir.mkdir(exist_ok=True)
        
        logger.info("Environment variables set for model caching")
        logger.info(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
        logger.info(f"TORCH_HOME: {os.environ['TORCH_HOME']}")
        logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
        
        # Download models
        logger.info("Starting model download (this may take 5-10 minutes)...")
        logger.info("Models being downloaded:")
        logger.info("- Layout detection model")
        logger.info("- OCR model (surya)")
        logger.info("- Text formatting model (texify)")
        logger.info("- Reading order model")
        
        models = create_model_dict()
        
        download_time = time.time() - start_time
        logger.info(f"=== MODEL DOWNLOAD COMPLETE ===")
        logger.info(f"Downloaded {len(models)} models in {download_time:.2f} seconds")
        logger.info(f"Models stored in: {target_dir}")
        
        # List downloaded files
        logger.info("=== DOWNLOADED FILES ===")
        for root, dirs, files in os.walk(target_dir):
            level = root.replace(target_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files in each dir
                file_size = os.path.getsize(os.path.join(root, file))
                logger.info(f"{subindent}{file} ({file_size // (1024*1024)}MB)")
            if len(files) > 5:
                logger.info(f"{subindent}... and {len(files) - 5} more files")
        
        return models
        
    except Exception as e:
        logger.error(f"❌ Model download failed: {str(e)}")
        raise
        
    finally:
        # Restore original environment variables
        if original_cache:
            os.environ['TRANSFORMERS_CACHE'] = original_cache
        else:
            os.environ.pop('TRANSFORMERS_CACHE', None)
            
        if original_torch_home:
            os.environ['TORCH_HOME'] = original_torch_home
        else:
            os.environ.pop('TORCH_HOME', None)
            
        if original_hf_home:
            os.environ['HF_HOME'] = original_hf_home
        else:
            os.environ.pop('HF_HOME', None)

def verify_models(model_dir: str):
    """Verify that models were downloaded correctly."""
    logger.info("=== VERIFYING DOWNLOADED MODELS ===")
    
    required_paths = [
        "models",
        "transformers",
    ]
    
    for path in required_paths:
        full_path = os.path.join(model_dir, path)
        if os.path.exists(full_path):
            logger.info(f"✓ Found: {path}")
        else:
            logger.warning(f"❌ Missing: {path}")
    
    # Check total size
    total_size = 0
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    
    total_size_gb = total_size / (1024**3)
    logger.info(f"Total downloaded size: {total_size_gb:.2f} GB")
    
    if total_size_gb < 0.5:
        logger.warning("Downloaded models seem small - download may be incomplete")
    else:
        logger.info("✓ Model download appears successful")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download marker models for AWS deployment")
    parser.add_argument(
        "--target-dir", 
        default="/opt/marker_models",
        help="Directory to store downloaded models (default: /opt/marker_models)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded models after download"
    )
    
    args = parser.parse_args()
    
    try:
        models = download_models_to_directory(args.target_dir)
        
        if args.verify:
            verify_models(args.target_dir)
            
        logger.info("=== SUCCESS ===")
        logger.info(f"Models ready at: {args.target_dir}")
        logger.info("You can now deploy to AWS with pre-downloaded models")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)
