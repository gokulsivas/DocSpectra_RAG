#!/usr/bin/env python3
"""
Memory-efficient marker model downloader for low-memory systems.
Downloads models one at a time with memory cleanup and verification.
Uses /opt/marker_models as default directory like the standard version.
"""

import os
import sys
import logging
import time
import gc
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_memory():
    """Check available memory."""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        for line in meminfo.split('\n'):
            if line.startswith('MemAvailable:'):
                available_kb = int(line.split()[1])
                available_gb = available_kb / (1024 * 1024)
                logger.info(f"Available memory: {available_gb:.2f} GB")
                return available_gb
    except:
        logger.warning("Could not check memory usage")
        return None

def download_models_sequentially(target_dir: str = "/opt/marker_models"):
    """
    Download marker models one at a time to minimize memory usage.
    
    Args:
        target_dir (str): Directory where models will be stored
    """
    target_dir = os.path.expanduser(target_dir)
    start_time = time.time()
    
    logger.info(f"=== MEMORY-EFFICIENT MODEL DOWNLOAD ===")
    logger.info(f"Target directory: {target_dir}")
    
    # Check initial memory
    check_memory()
    
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
        
        logger.info("Environment variables configured for model caching")
        logger.info(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
        logger.info(f"TORCH_HOME: {os.environ['TORCH_HOME']}")
        logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
        
        # Import marker components individually to control memory usage
        logger.info("=== DOWNLOADING MODELS SEQUENTIALLY ===")
        logger.info("Models being downloaded:")
        logger.info("- Layout detection model")
        logger.info("- OCR model (surya)")
        logger.info("- Text formatting model (texify)")
        logger.info("- Reading order model")
        
        # Step 1: Download layout detection model
        logger.info("Step 1/4: Downloading layout detection model...")
        try:
            from marker.models import load_layout_model
            layout_model = load_layout_model()
            logger.info("✓ Layout model downloaded")
            
            # Clean up
            del layout_model
            gc.collect()
            check_memory()
            
        except ImportError:
            logger.info("Using alternative method for layout model...")
            from marker.models import create_model_dict
            models = create_model_dict()
            logger.info("✓ All models downloaded via create_model_dict()")
            
            download_time = time.time() - start_time
            logger.info(f"=== DOWNLOAD COMPLETE ===")
            logger.info(f"Downloaded {len(models)} models in {download_time:.2f} seconds")
            logger.info(f"Models stored in: {target_dir}")
            return models
        
        # Step 2: Download OCR model  
        logger.info("Step 2/4: Downloading OCR model...")
        try:
            from marker.models import load_ocr_model
            ocr_model = load_ocr_model()
            logger.info("✓ OCR model downloaded")
            
            # Clean up
            del ocr_model
            gc.collect()
            check_memory()
            
        except ImportError:
            logger.info("OCR model included in main download")
        
        # Step 3: Download text formatting model
        logger.info("Step 3/4: Downloading text formatting model...")
        try:
            from marker.models import load_texify_model
            texify_model = load_texify_model()
            logger.info("✓ Text formatting model downloaded")
            
            # Clean up
            del texify_model
            gc.collect()
            check_memory()
            
        except ImportError:
            logger.info("Text formatting model included in main download")
        
        # Step 4: Download reading order model
        logger.info("Step 4/4: Downloading reading order model...")
        try:
            from marker.models import load_order_model
            order_model = load_order_model()
            logger.info("✓ Reading order model downloaded")
            
            # Clean up
            del order_model
            gc.collect()
            check_memory()
            
        except ImportError:
            logger.info("Reading order model included in main download")
        
        # Final step: Load all models together (they should be cached now)
        logger.info("Loading all models from cache...")
        from marker.models import create_model_dict
        models = create_model_dict()
        
        download_time = time.time() - start_time
        logger.info(f"=== DOWNLOAD COMPLETE ===")
        logger.info(f"Downloaded {len(models)} models in {download_time:.2f} seconds")
        logger.info(f"Models stored in: {target_dir}")
        
        # List downloaded files
        logger.info("=== DOWNLOADED FILES ===")
        list_downloaded_files(target_dir)
        
        return models
        
    except Exception as e:
        logger.error(f"❌ Model download failed: {str(e)}")
        
        # Check if it's an OOM error
        if "killed" in str(e).lower() or "memory" in str(e).lower():
            logger.error("=== OUT OF MEMORY ERROR ===")
            logger.error("Your system doesn't have enough RAM for marker models")
            logger.error("Solutions:")
            logger.error("1. Upgrade to EC2 instance with more memory (t3.medium or larger)")
            logger.error("2. Add swap space: sudo fallocate -l 4G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile")
            logger.error("3. Use a machine with at least 4GB RAM")
        
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

def list_downloaded_files(target_dir: str):
    """List downloaded files with sizes."""
    for root, dirs, files in os.walk(target_dir):
        level = root.replace(target_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        logger.info(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files in each dir
            try:
                file_size = os.path.getsize(os.path.join(root, file))
                logger.info(f"{subindent}{file} ({file_size // (1024*1024)}MB)")
            except:
                logger.info(f"{subindent}{file} (size unknown)")
        if len(files) > 5:
            logger.info(f"{subindent}... and {len(files) - 5} more files")

def verify_models(model_dir: str):
    """Verify that models were downloaded correctly."""
    model_dir = os.path.expanduser(model_dir)
    logger.info("=== VERIFYING DOWNLOADED MODELS ===")
    
    if not os.path.exists(model_dir):
        logger.error(f"❌ Model directory not found: {model_dir}")
        return False
    
    # Check for expected directories
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
    file_count = 0
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                file_count += 1
            except:
                pass
    
    total_size_gb = total_size / (1024**3)
    logger.info(f"Total downloaded: {file_count} files, {total_size_gb:.2f} GB")
    
    if total_size_gb < 0.5:
        logger.warning("❌ Downloaded models seem too small - download may be incomplete")
        return False
    else:
        logger.info("✓ Model download appears successful")
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory-efficient marker model downloader for AWS deployment")
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
    
    # Check system memory first
    available_memory = check_memory()
    if available_memory and available_memory < 2.0:
        logger.warning(f"⚠️  Low memory detected: {available_memory:.2f} GB")
        logger.warning("Marker models require ~3-4GB RAM. Consider:")
        logger.warning("1. Upgrading to a larger EC2 instance")
        logger.warning("2. Adding swap space")
        logger.warning("3. Closing other applications")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    try:
        models = download_models_sequentially(args.target_dir)
        
        if args.verify:
            verify_models(args.target_dir)
            
        logger.info("=== SUCCESS ===")
        logger.info(f"Models ready at: {args.target_dir}")
        logger.info("You can now deploy to AWS with pre-downloaded models")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)
