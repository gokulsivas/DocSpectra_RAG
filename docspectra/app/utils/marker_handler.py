# marker.py - Enhanced version with URL download support
import os
import logging
import time
import subprocess
import sys
import requests
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path
from urllib.parse import urlparse
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

# Set up verbose logging for AWS CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration for pre-downloaded models
MODEL_BASE_PATH = os.getenv('MARKER_MODEL_PATH', '/opt/marker_models')
USE_LOCAL_MODELS = os.getenv('USE_LOCAL_MODELS', 'true').lower() == 'true'
S3_BUCKET = os.getenv('S3_MODELS_BUCKET', 's3://markerbucket69/marker_models/')
AUTO_DOWNLOAD = os.getenv('AUTO_DOWNLOAD_MODELS', 'true').lower() == 'true'

# Global variables for model caching
_models = None
_converter = None
_initialization_time = None

class S3ModelManager:
    """Manages model downloads and caching from S3"""
    
    def __init__(self):
        self.model_base_path = MODEL_BASE_PATH
        self.s3_bucket = S3_BUCKET
        self.auto_download = AUTO_DOWNLOAD
    
    def check_aws_cli(self) -> bool:
        """Check if AWS CLI is available and configured."""
        try:
            result = subprocess.run(['aws', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✓ AWS CLI available: {result.stdout.strip()}")
                return True
            else:
                logger.error("❌ AWS CLI not working properly")
                return False
        except FileNotFoundError:
            logger.error("❌ AWS CLI not installed")
            return False
        except subprocess.TimeoutExpired:
            logger.error("❌ AWS CLI command timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error checking AWS CLI: {e}")
            return False
    
    def download_models_from_s3(self) -> bool:
        """Download models from S3 bucket if they don't exist locally."""
        if not self.auto_download:
            logger.info("Auto-download disabled, skipping S3 download")
            return False

        logger.info("=== DOWNLOADING MODELS FROM S3 ===")
        logger.info(f"Source: {self.s3_bucket}")
        logger.info(f"Destination: {self.model_base_path}")

        # Check if AWS CLI is available
        if not self.check_aws_cli():
            logger.error("Cannot download from S3 - AWS CLI not available")
            return False

        # Create directory if it doesn't exist
        try:
            os.makedirs(self.model_base_path, exist_ok=True)
            logger.info(f"✓ Created/verified directory: {self.model_base_path}")
        except Exception as e:
            logger.error(f"❌ Failed to create directory {self.model_base_path}: {e}")
            return False

        # Download models using AWS S3 CP
        download_start = time.time()
        try:
            logger.info("Starting S3 download...")
            cmd = [
                'aws', 's3', 'cp', '--recursive',
                self.s3_bucket.rstrip('/') + '/',
                self.model_base_path + '/',
                '--no-follow-symlinks'
            ]
            logger.info(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            download_time = time.time() - download_start

            if result.returncode == 0:
                logger.info(f"✅ S3 download completed in {download_time:.2f} seconds")
                if result.stdout:
                    logger.info(f"AWS output: {result.stdout.strip()}")
                
                if self.verify_downloaded_models():
                    logger.info("✅ Model download verification passed")
                    return True
                else:
                    logger.error("❌ Model download verification failed")
                    return False
            else:
                logger.error(f"❌ S3 download failed after {download_time:.2f}s")
                logger.error(f"Return code: {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr.strip()}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ S3 download timed out after 1 hour")
            return False
        except Exception as e:
            download_time = time.time() - download_start
            logger.error(f"❌ S3 download failed after {download_time:.2f}s: {e}")
            return False

    def verify_downloaded_models(self) -> bool:
        """Verify that models were downloaded successfully."""
        if not os.path.exists(self.model_base_path):
            logger.error(f"❌ Model directory not found: {self.model_base_path}")
            return False

        try:
            total_size = 0
            file_count = 0
            model_files_found = False

            for root, dirs, files in os.walk(self.model_base_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        file_count += 1
                        # Check for model file types
                        if file.endswith(('.bin', '.safetensors', '.pt', '.pth', '.json')):
                            model_files_found = True
                    except OSError:
                        continue

            total_size_gb = total_size / (1024**3)
            logger.info(f"Downloaded models: {file_count} files, {total_size_gb:.2f} GB")

            if file_count == 0:
                logger.error("❌ No files found in model directory")
                return False

            if not model_files_found:
                logger.warning("⚠️ No model files (.bin, .safetensors, .pt, .json) found")

            if total_size_gb < 0.05:
                logger.warning(f"⚠️ Total model size ({total_size_gb:.2f} GB) seems small")

            return True

        except Exception as e:
            logger.error(f"❌ Error verifying models: {e}")
            return False

    def check_models_exist_locally(self) -> bool:
        """Check if models exist locally."""
        return os.path.exists(self.model_base_path) and os.listdir(self.model_base_path)

    def ensure_models_available(self) -> bool:
        """Ensure models are available locally, download if needed."""
        if self.check_models_exist_locally():
            return True
        
        if self.auto_download:
            return self.download_models_from_s3()
        
        return False

    def initialize_and_cache_models(self):
        """Initialize models and cache to S3 if needed."""
        # First ensure models are available
        if not self.ensure_models_available():
            logger.warning("Models not available locally and download failed")
        
        # Setup environment for local models
        self.setup_model_paths()
        
        # Create and return model dict
        return create_model_dict()

    def setup_model_paths(self):
        """Configure environment to use pre-downloaded models."""
        if not USE_LOCAL_MODELS:
            logger.info("Local models disabled, will use internet download")
            return False

        logger.info(f"=== CONFIGURING LOCAL MODEL PATHS ===")
        logger.info(f"Model base path: {self.model_base_path}")

        # Set environment variables to point to our pre-downloaded models
        os.environ['HF_HOME'] = self.model_base_path
        os.environ['TORCH_HOME'] = self.model_base_path

        # Also set Hugging Face hub cache if it exists
        hf_cache = os.path.join(self.model_base_path, 'hub')
        if os.path.exists(hf_cache):
            os.environ['HF_HUB_CACHE'] = hf_cache
            logger.info(f"✓ Set HF_HUB_CACHE: {hf_cache}")

        logger.info("✓ Environment configured for local models")
        logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
        logger.info(f"TORCH_HOME: {os.environ['TORCH_HOME']}")

        return True


def download_pdf_from_url(url: str, timeout: int = 30) -> Optional[str]:
    """
    Download PDF from URL to a temporary file.
    
    Args:
        url (str): URL to download PDF from
        timeout (int): Request timeout in seconds
        
    Returns:
        Optional[str]: Path to downloaded temporary file, or None if failed
    """
    try:
        logger.info(f"Downloading PDF from URL: {url}")
        
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.error(f"Invalid URL format: {url}")
            return None
        
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        download_start = time.time()
        
        # Download with streaming to handle large files
        response = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        # Check if content is actually a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            logger.warning(f"Content type '{content_type}' may not be PDF")
        
        # Create temporary file with PDF extension
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_path = temp_file.name
        
        # Download and write to temp file
        total_size = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
                total_size += len(chunk)
        
        temp_file.close()
        download_time = time.time() - download_start
        
        # Verify the download
        if os.path.getsize(temp_path) == 0:
            logger.error("Downloaded file is empty")
            os.unlink(temp_path)
            return None
        
        total_size_mb = total_size / (1024 * 1024)
        logger.info(f"✅ PDF downloaded successfully in {download_time:.2f}s ({total_size_mb:.2f} MB): {temp_path}")
        
        return temp_path
        
    except requests.exceptions.Timeout:
        logger.error(f"❌ Download timeout after {timeout}s for URL: {url}")
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Connection error downloading from URL: {url}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ HTTP error {e.response.status_code} downloading from URL: {url}")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Request error downloading from URL: {url} - {str(e)}")
    except Exception as e:
        logger.error(f"❌ Unexpected error downloading PDF: {str(e)}")
    
    return None


def cleanup_temp_file(file_path: str):
    """Safely cleanup temporary file."""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")


class MarkerHandler:
    """Enhanced Marker handler with S3 model management and URL support"""
    
    def __init__(self, use_llm=False, output_format='markdown'):
        self.use_llm = use_llm
        self.output_format = output_format
        self.model_manager = S3ModelManager()
        self.converter = None
        self._initialize_converter()

    def _initialize_converter(self):
        """Initialize the Marker converter with S3 model management"""
        try:
            # Ensure models are available locally
            models_ready = self.model_manager.ensure_models_available()
            
            if not models_ready:
                # Download fresh models and cache to S3
                logger.info("Downloading fresh models...")
                model_dict = self.model_manager.initialize_and_cache_models()
            else:
                # Use existing cached models
                self.model_manager.setup_model_paths()
                model_dict = create_model_dict()

            # Initialize converter
            self.converter = PdfConverter(artifact_dict=model_dict)
            
            logger.info("Marker converter initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Marker converter: {e}")
            raise

    def convert_pdf(self, pdf_input, page_range=None):
        """
        Convert PDF to markdown/json - supports both local files and URLs
        
        Args:
            pdf_input (str): Path to PDF file or URL to PDF
            page_range (str, optional): Page range like "0,5-10,20"
            
        Returns:
            dict: Contains text, metadata, and images
        """
        if not self.converter:
            raise RuntimeError("Marker converter not initialized")

        temp_file_path = None
        try:
            # Determine if input is URL or local file
            if pdf_input.startswith(('http://', 'https://')):
                logger.info(f"Processing PDF from URL: {pdf_input}")
                # Download PDF from URL
                temp_file_path = download_pdf_from_url(pdf_input)
                if not temp_file_path:
                    raise RuntimeError(f"Failed to download PDF from URL: {pdf_input}")
                pdf_path = Path(temp_file_path)
                source_identifier = pdf_input  # Use URL as source identifier
            else:
                # Local file path
                pdf_path = Path(pdf_input)
                if not pdf_path.exists():
                    raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                source_identifier = str(pdf_path)

            logger.info(f"Converting PDF: {pdf_path}")

            # Convert PDF
            rendered = self.converter(str(pdf_path))

            # Extract text, metadata, and images
            text, metadata, images = text_from_rendered(rendered)

            result = {
                'text': text,
                'metadata': metadata,
                'images': images,
                'source_file': source_identifier,
                'temp_file': temp_file_path  # Include temp file path for cleanup
            }

            logger.info(f"Successfully converted PDF from: {source_identifier}")
            return result

        except Exception as e:
            # Cleanup temp file on error
            if temp_file_path:
                cleanup_temp_file(temp_file_path)
            logger.error(f"Error converting PDF {pdf_input}: {e}")
            raise

    def convert_pdf_batch(self, pdf_inputs):
        """
        Convert multiple PDFs in batch - supports both local files and URLs
        
        Args:
            pdf_inputs (list): List of PDF file paths or URLs
            
        Returns:
            list: List of conversion results
        """
        results = []
        for pdf_input in pdf_inputs:
            try:
                result = self.convert_pdf(pdf_input)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to convert {pdf_input}: {e}")
                results.append({
                    'error': str(e),
                    'source_file': str(pdf_input)
                })
        
        return results

    def health_check(self):
        """Check if the converter is working properly"""
        try:
            return {
                'status': 'healthy',
                'converter_initialized': self.converter is not None,
                'models_available': self.model_manager.check_models_exist_locally(),
                'use_llm': self.use_llm,
                'output_format': self.output_format,
                'supports_urls': True
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


# Legacy functions for backward compatibility with URL support
def initialize_models():
    """Initialize models using S3 model management."""
    global _models, _converter, _initialization_time
    
    if _models is not None:
        logger.info("Models already initialized")
        return True

    start_time = time.time()
    logger.info("=== INITIALIZING MARKER MODELS ===")

    try:
        # Use S3ModelManager for model handling
        model_manager = S3ModelManager()
        
        # Setup model paths (includes S3 download if needed)
        local_models_available = model_manager.setup_model_paths()
        
        if local_models_available:
            if not model_manager.check_models_exist_locally():
                logger.error("Local model verification failed")
                logger.info("Attempting to download from S3...")
                if not model_manager.download_models_from_s3():
                    logger.info("Falling back to internet download...")
                    local_models_available = False
            else:
                logger.info("Loading models from local storage...")
        
        if not local_models_available:
            logger.info("Loading models from internet (this will be slow)...")

        # Load models
        logger.info("Creating model dictionary...")
        _models = create_model_dict()
        logger.info(f"✓ Loaded {len(_models)} models")

        # Create converter
        logger.info("Creating PDF converter...")
        _converter = PdfConverter(artifact_dict=_models)
        logger.info("✓ PDF converter ready")

        _initialization_time = time.time() - start_time
        logger.info(f"=== INITIALIZATION COMPLETE ===")
        logger.info(f"Total time: {_initialization_time:.2f} seconds")
        logger.info(f"Model source: {'Local storage' if local_models_available else 'Internet download'}")

        return True

    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"❌ Model initialization failed after {error_time:.2f}s: {str(e)}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def parse_pdf_from_url(url: str) -> Optional[str]:
    """Parse PDF from URL using pre-loaded models - Enhanced with proper URL handling."""
    request_start = time.time()
    logger.info(f"=== NEW PDF PARSING REQUEST ===")
    logger.info(f"Request URL: {url}")

    # Input validation
    if not url or not isinstance(url, str):
        logger.error("Invalid URL provided")
        return None

    # Ensure models are initialized
    if not initialize_models():
        logger.error("❌ Cannot process request - model initialization failed")
        return None

    temp_file_path = None
    try:
        # URL preprocessing
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
            logger.info(f"Added HTTPS protocol: {url}")

        logger.info(f"Processing PDF from: {url}")
        logger.info("✓ Using pre-loaded models (zero load time)")

        # Download PDF from URL
        download_start = time.time()
        logger.info("Downloading PDF from URL...")
        temp_file_path = download_pdf_from_url(url)
        
        if not temp_file_path:
            logger.error("❌ Failed to download PDF from URL")
            return None
            
        download_time = time.time() - download_start
        logger.info(f"✓ PDF downloaded in {download_time:.2f}s")

        # PDF conversion
        conversion_start = time.time()
        logger.info("Starting PDF conversion...")
        rendered = _converter(temp_file_path)  # Use local temp file
        conversion_time = time.time() - conversion_start
        logger.info(f"✓ Conversion completed in {conversion_time:.2f}s")

        # Text extraction
        extraction_start = time.time()
        logger.info("Extracting text...")
        text, metadata, images = text_from_rendered(rendered)
        extraction_time = time.time() - extraction_start
        logger.info(f"✓ Extraction completed in {extraction_time:.2f}s")

        # Results
        if text:
            char_count = len(text)
            word_count = len(text.split())
            logger.info(f"=== SUCCESS ===")
            logger.info(f"✓ Text: {char_count} chars, {word_count} words")
            logger.info(f"✓ Images: {len(images) if images else 0}")
        else:
            logger.warning("⚠️ No text extracted from PDF")

        # Performance summary
        total_time = time.time() - request_start
        logger.info(f"=== REQUEST COMPLETE ===")
        logger.info(f"Total: {total_time:.2f}s (Download: {download_time:.2f}s, Conv: {conversion_time:.2f}s, Ext: {extraction_time:.2f}s)")

        return text if text else None

    except Exception as e:
        error_time = time.time() - request_start
        logger.error(f"=== PARSING FAILED ===")
        logger.error(f"❌ Error after {error_time:.2f}s: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    
    finally:
        # Always cleanup temp file
        if temp_file_path:
            cleanup_temp_file(temp_file_path)


# Additional legacy wrapper functions
def setup_model_paths():
    """Legacy function - now uses S3ModelManager"""
    manager = S3ModelManager()
    return manager.setup_model_paths()


def verify_local_models():
    """Legacy function - now uses S3ModelManager"""
    manager = S3ModelManager()
    return manager.check_models_exist_locally()


def download_models_from_s3():
    """Legacy function - now uses S3ModelManager"""
    manager = S3ModelManager()
    return manager.download_models_from_s3()


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging."""
    model_manager = S3ModelManager()
    
    info = {
        'model_path': MODEL_BASE_PATH,
        'use_local_models': USE_LOCAL_MODELS,
        's3_bucket': S3_BUCKET,
        'auto_download': AUTO_DOWNLOAD,
        'models_initialized': _models is not None,
        'converter_ready': _converter is not None,
        'initialization_time': _initialization_time,
        'local_models_exist': model_manager.check_models_exist_locally(),
        'aws_cli_available': model_manager.check_aws_cli(),
        'supports_urls': True
    }

    if USE_LOCAL_MODELS and os.path.exists(MODEL_BASE_PATH):
        # Get model directory size
        total_size = 0
        for root, dirs, files in os.walk(MODEL_BASE_PATH):
            for file in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, file))
                except:
                    pass
        info['model_size_gb'] = total_size / (1024**3)

    logger.info(f"System info: {info}")
    return info


def force_download_models() -> bool:
    """Force download models from S3 (for manual triggering)."""
    logger.info("=== FORCE DOWNLOADING MODELS ===")
    
    # Remove existing models if they exist
    if os.path.exists(MODEL_BASE_PATH):
        import shutil
        logger.info(f"Removing existing models at {MODEL_BASE_PATH}")
        shutil.rmtree(MODEL_BASE_PATH)

    # Download fresh models
    model_manager = S3ModelManager()
    return model_manager.download_models_from_s3()


# Initialize on import for AWS Lambda warm containers
logger.info("=== MARKER PDF PARSER STARTING ===")
logger.info(f"Model path: {MODEL_BASE_PATH}")
logger.info(f"Use local models: {USE_LOCAL_MODELS}")
logger.info(f"S3 bucket: {S3_BUCKET}")
logger.info(f"Auto download: {AUTO_DOWNLOAD}")

# Try to initialize immediately for warm starts
if USE_LOCAL_MODELS:
    logger.info("Attempting immediate model initialization...")
    try:
        initialize_models()
    except Exception as e:
        logger.warning(f"Initial model loading failed: {e}")
        logger.info("Will retry on first request")

logger.info("=== MARKER PDF PARSER READY ===")
