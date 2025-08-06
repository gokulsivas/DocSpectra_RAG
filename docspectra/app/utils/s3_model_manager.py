# app/utils/s3_model_manager.py
import os
import boto3
import subprocess
import logging
from pathlib import Path
from marker.models import create_model_dict

logger = logging.getLogger(__name__)

class S3ModelManager:
    def __init__(self, bucket_name='markerbucket69', region_name='us-east-1'):
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.cache_dir = Path.home() / '.cache'
        
    def check_models_exist_locally(self):
        """Check if models exist in local cache"""
        model_dirs = ['surya', 'texify', 'marker']
        existing_models = {}
        
        for model_dir in model_dirs:
            local_path = self.cache_dir / model_dir
            existing_models[model_dir] = local_path.exists() and any(local_path.iterdir())
            
        return existing_models
    
    def upload_models_to_s3(self):
        """Upload local models to S3 bucket"""
        model_dirs = ['surya', 'texify', 'marker']
        
        for model_dir in model_dirs:
            local_path = self.cache_dir / model_dir
            if local_path.exists():
                try:
                    result = subprocess.run([
                        'aws', 's3', 'sync', 
                        str(local_path), 
                        f's3://{self.bucket_name}/{model_dir}/',
                        '--region', self.region_name
                    ], check=True, capture_output=True, text=True)
                    logger.info(f"Successfully uploaded {model_dir} models to S3")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to upload {model_dir} models: {e.stderr}")
    
    def download_models_from_s3(self):
        """Download models from S3 to local cache"""
        model_dirs = ['surya', 'texify', 'marker']
        
        for model_dir in model_dirs:
            local_path = self.cache_dir / model_dir
            local_path.mkdir(parents=True, exist_ok=True)
            
            try:
                result = subprocess.run([
                    'aws', 's3', 'sync',
                    f's3://{self.bucket_name}/{model_dir}/',
                    str(local_path),
                    '--region', self.region_name
                ], check=True, capture_output=True, text=True)
                logger.info(f"Successfully downloaded {model_dir} models from S3")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Could not download {model_dir} models from S3: {e.stderr}")
    
    def ensure_models_available(self):
        """Ensure models are available locally, download from S3 if needed"""
        existing_models = self.check_models_exist_locally()
        
        if not all(existing_models.values()):
            logger.info("Some models missing locally, attempting to download from S3...")
            self.download_models_from_s3()
            
            # Check again after download
            existing_models = self.check_models_exist_locally()
            
            if not all(existing_models.values()):
                logger.info("Models not found in S3, will download fresh models...")
                return False
        
        return True
    
    def initialize_and_cache_models(self):
        """Initialize models and cache them to S3"""
        logger.info("Initializing Marker models...")
        
        # This will download models if they don't exist locally
        model_dict = create_model_dict()
        
        # Upload to S3 for future use
        self.upload_models_to_s3()
        
        return model_dict
