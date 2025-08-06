# setup_models.py
"""
Script to download Marker models and upload them to S3 bucket
Run this once to populate your S3 bucket with models
"""

import os
import sys
import logging
from pathlib import Path

# Add app directory to Python path
sys.path.append(str(Path(__file__).parent / 'app'))

from utils.s3_model_manager import S3ModelManager
from marker.models import create_model_dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main setup function"""
    logger.info("Starting Marker models setup for S3...")
    
    # Initialize S3 model manager
    model_manager = S3ModelManager(bucket_name='markerbucket69', region_name='us-east-1')
    
    # Check if models already exist locally
    existing_models = model_manager.check_models_exist_locally()
    logger.info(f"Local models status: {existing_models}")
    
    # Check if models exist in S3
    try:
        # Try to list objects in S3 bucket
        response = model_manager.s3_client.list_objects_v2(
            Bucket='markerbucket69',
            Prefix='surya/',
            MaxKeys=1
        )
        models_in_s3 = response.get('KeyCount', 0) > 0
        logger.info(f"Models found in S3: {models_in_s3}")
    except Exception as e:
        logger.warning(f"Could not check S3 bucket: {e}")
        models_in_s3 = False
    
    if not any(existing_models.values()) and not models_in_s3:
        logger.info("No models found locally or in S3. Downloading fresh models...")
        
        # Download models for the first time
        logger.info("This may take several minutes...")
        model_dict = create_model_dict()
        logger.info("Models downloaded successfully!")
        
        # Upload to S3
        logger.info("Uploading models to S3...")
        model_manager.upload_models_to_s3()
        logger.info("Models uploaded to S3 successfully!")
        
    elif models_in_s3 and not all(existing_models.values()):
        logger.info("Models found in S3 but not locally. Downloading from S3...")
        model_manager.download_models_from_s3()
        
    elif all(existing_models.values()) and not models_in_s3:
        logger.info("Models found locally but not in S3. Uploading to S3...")
        model_manager.upload_models_to_s3()
        
    else:
        logger.info("Models are already available both locally and in S3!")
    
    # Verify setup
    final_local_status = model_manager.check_models_exist_locally()
    logger.info(f"Final local models status: {final_local_status}")
    
    # Test model loading
    try:
        logger.info("Testing model loading...")
        test_dict = create_model_dict()
        logger.info("✅ Model loading test successful!")
        
        # Print model info
        for key, value in test_dict.items():
            if hasattr(value, 'config'):
                logger.info(f"Model {key}: {type(value).__name__}")
            else:
                logger.info(f"Artifact {key}: {type(value).__name__}")
                
    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        return False
    
    logger.info("🎉 Setup completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
