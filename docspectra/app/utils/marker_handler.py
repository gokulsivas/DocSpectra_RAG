import os
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import boto3
from botocore.exceptions import ClientError
import threading
from functools import lru_cache
import requests
from urllib.parse import urlparse

# Modern Marker imports
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3ModelManager:
    """Manages downloading and caching models from S3 for Marker"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.region = region
        self.cache_dir = tempfile.mkdtemp(prefix='marker_models_')
        self._cache_lock = threading.Lock()
        self._downloaded_models = set()
        
        # Initialize S3 client (using IAM role, no explicit credentials needed)
        self.s3_client = boto3.client('s3', region_name=region)
        
        # Model mappings that Marker expects
        self.model_mappings = {
            'layout_detection': 'LAYOUT_MODEL_CHECKPOINT',
            'text_detection': 'DETECTOR_MODEL_CHECKPOINT', 
            'text_recognition': 'RECOGNITION_MODEL_CHECKPOINT',
            'table_recognition': 'TABLE_REC_MODEL_CHECKPOINT',
            'texify': 'TEXIFY_MODEL_CHECKPOINT'
        }
        
        logger.info(f"Initialized S3ModelManager with temp dir: {self.cache_dir}")
    
    def _download_model_from_s3(self, model_name: str) -> str:
        """Download model from S3 to local cache"""
        local_model_path = os.path.join(self.cache_dir, model_name)
        s3_prefix = f"models/{model_name}/"
        
        if model_name in self._downloaded_models:
            logger.info(f"Model {model_name} already cached locally")
            return local_model_path
        
        logger.info(f"Downloading {model_name} from S3 to {local_model_path}")
        
        # Create local directory
        os.makedirs(local_model_path, exist_ok=True)
        
        try:
            # List all objects with the model prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=s3_prefix
            )
            
            if 'Contents' not in response:
                raise ValueError(f"No files found for model {model_name} in S3")
            
            # Download all model files
            for obj in response['Contents']:
                s3_key = obj['Key']
                if s3_key.endswith('/'):  # Skip directory markers
                    continue
                
                # Calculate local file path
                relative_path = s3_key[len(s3_prefix):]
                local_file_path = os.path.join(local_model_path, relative_path)
                
                # Create subdirectories if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Download file
                self.s3_client.download_file(self.bucket_name, s3_key, local_file_path)
                logger.debug(f"Downloaded {s3_key} -> {local_file_path}")
            
            self._downloaded_models.add(model_name)
            logger.info(f"Successfully downloaded {model_name}")
            return local_model_path
            
        except ClientError as e:
            logger.error(f"Error downloading {model_name}: {str(e)}")
            raise
    
    def setup_model_environment(self, models_to_load: Optional[List[str]] = None):
        """Setup environment variables pointing to cached model paths"""
        if models_to_load is None:
            models_to_load = list(self.model_mappings.keys())
        
        for model_name in models_to_load:
            if model_name not in self.model_mappings:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            try:
                # Download model and get local path
                local_path = self._download_model_from_s3(model_name)
                
                # Set environment variable for marker
                env_var = self.model_mappings[model_name]
                os.environ[env_var] = local_path
                
                logger.info(f"Set {env_var} = {local_path}")
                
            except Exception as e:
                logger.error(f"Failed to setup {model_name}: {str(e)}")
                raise
        
        # Set cache directory for additional models
        os.environ['MODEL_CACHE_DIR'] = self.cache_dir
        logger.info(f"Set MODEL_CACHE_DIR = {self.cache_dir}")
    
    def list_available_models(self) -> Dict[str, bool]:
        """Check which models are available in S3"""
        available = {}
        
        for model_name in self.model_mappings.keys():
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=f"models/{model_name}/",
                    MaxKeys=1
                )
                available[model_name] = 'Contents' in response
            except Exception:
                available[model_name] = False
        
        return available
    
    def cleanup(self):
        """Clean up temporary cache directory"""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleaned up model cache directory: {self.cache_dir}")

class MarkerHandlerWithS3Models:
    """
    Marker handler that loads models from S3 and processes documents from URLs only.
    No S3 document processing - only model loading from S3.
    """
    
    def __init__(self, 
                 model_bucket: str,
                 aws_region: str = "us-east-1",
                 use_llm: bool = False,
                 batch_multiplier: int = 1,
                 download_timeout: int = 300):
        """
        Initialize handler with S3 models.
        
        Args:
            model_bucket: S3 bucket containing Marker models (NOT documents)
            aws_region: AWS region
            use_llm: Whether to use LLM for enhanced accuracy
            batch_multiplier: Batch size multiplier for processing
            download_timeout: URL download timeout in seconds
        """
        self.model_bucket = model_bucket
        self.aws_region = aws_region
        self.use_llm = use_llm
        self.batch_multiplier = batch_multiplier
        self.download_timeout = download_timeout
        
        # Initialize S3 model manager
        self.model_manager = S3ModelManager(model_bucket, aws_region)
        
        # Marker components (loaded on demand)
        self._model_dict = None
        self._converter = None
        self._models_loaded = False
        self._cache_lock = threading.Lock()
        
        # Document cache for URL downloads
        self.document_cache_dir = tempfile.mkdtemp(prefix='marker_documents_')
        self._temp_files = set()
        
        logger.info(f"Initialized MarkerHandlerWithS3Models")
        logger.info(f"Model bucket: {model_bucket}")
        logger.info(f"Document processing: URLs only (no S3 documents)")

    def _setup_marker_environment(self):
        """Setup environment for optimal Marker performance"""
        # Setup models from S3
        self.model_manager.setup_model_environment()
        
        # Additional Marker configuration
        os.environ.setdefault("TORCH_DEVICE", "cuda" if self._has_gpu() else "cpu")
        os.environ.setdefault("OCR_ENGINE", "surya")
        os.environ.setdefault("EXTRACT_IMAGES", "true")
        
        if self.batch_multiplier > 1:
            os.environ["BATCH_MULTIPLIER"] = str(self.batch_multiplier)

    def _has_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _ensure_models_loaded(self):
        """Load Marker models with S3-downloaded model checkpoints"""
        if not self._models_loaded:
            with self._cache_lock:
                if not self._models_loaded:  # Double-check locking
                    logger.info("Setting up Marker models from S3...")
                    
                    # Setup environment with S3 models
                    self._setup_marker_environment()
                    
                    # Create model dictionary with our S3 models
                    self._model_dict = create_model_dict()
                    
                    # Configure Marker
                    config = {
                        "output_format": "markdown",
                        "extract_images": True,
                    }
                    
                    if self.use_llm:
                        config["use_llm"] = True
                        config["llm_model"] = "gemini-2.0-flash"
                    
                    config_parser = ConfigParser(config)
                    
                    # Initialize converter
                    self._converter = PdfConverter(
                        config=config_parser.generate_config_dict(),
                        artifact_dict=self._model_dict,
                        processor_list=config_parser.get_processors(),
                        renderer=config_parser.get_renderer(),
                        llm_service=config_parser.get_llm_service() if self.use_llm else None
                    )
                    
                    self._models_loaded = True
                    logger.info("Marker models loaded successfully from S3")

    def download_document_from_url(self, url: str) -> str:
        """Download document from URL to temporary local file"""
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {url}")
            
            logger.info(f"Downloading document from URL: {url}")
            
            # Configure session
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'MarkerS3Models/1.0 (Document Processing)',
                'Accept': 'application/pdf,application/octet-stream,*/*'
            })
            
            # Download with streaming
            response = session.get(url, timeout=self.download_timeout, stream=True)
            response.raise_for_status()
            
            # Determine file extension
            content_type = response.headers.get('content-type', '').lower()
            file_extension = '.pdf'  # Default
            
            if 'pdf' in content_type:
                file_extension = '.pdf'
            else:
                url_path = parsed_url.path
                if url_path:
                    _, ext = os.path.splitext(url_path)
                    if ext.lower() in ['.pdf', '.doc', '.docx']:
                        file_extension = ext.lower()
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                dir=self.document_cache_dir,
                suffix=file_extension,
                delete=False
            )
            
            # Download in chunks
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with temp_file as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0 and downloaded_size % (1024 * 1024) == 0:
                            progress = (downloaded_size / total_size) * 100
                            logger.debug(f"Download progress: {progress:.1f}%")
            
            # Validate file
            file_size = os.path.getsize(temp_file.name)
            if file_size == 0:
                os.unlink(temp_file.name)
                raise ValueError("Downloaded file is empty")
            
            # Basic PDF validation
            if file_extension == '.pdf':
                with open(temp_file.name, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        logger.warning("Downloaded file may not be a valid PDF")
            
            self._temp_files.add(temp_file.name)
            logger.info(f"Successfully downloaded to: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error downloading from {url}: {str(e)}")
            raise

    def convert_pdf_to_markdown(self, pdf_path: str, **kwargs) -> Tuple[str, Dict, List]:
        """Convert PDF to markdown using S3 models"""
        try:
            # Ensure S3 models are loaded
            self._ensure_models_loaded()
            
            logger.info(f"Converting PDF: {pdf_path}")
            
            # Convert PDF using Marker with S3 models
            rendered = self._converter(pdf_path)
            markdown_text, metadata, images = text_from_rendered(rendered)
            
            return markdown_text, metadata, images
            
        except Exception as e:
            logger.error(f"Error converting PDF: {str(e)}")
            raise

    def process_document_from_url(self, 
                                document_url: str,
                                questions: Optional[List[str]] = None,
                                **kwargs) -> Dict[str, Any]:
        """Process document from URL using S3 models"""
        temp_document_path = None
        
        try:
            # Download document
            temp_document_path = self.download_document_from_url(document_url)
            
            # Convert to markdown
            markdown_text, metadata, images = self.convert_pdf_to_markdown(
                temp_document_path, **kwargs
            )
            
            result = {
                'success': True,
                'text': markdown_text,
                'images': images,
                'metadata': metadata,
                'source_url': document_url
            }
            
            # Answer questions if provided
            if questions:
                result['qa_results'] = self._generate_answers(markdown_text, questions)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document from URL {document_url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'source_url': document_url
            }

    def _generate_answers(self, document_text: str, questions: List[str]) -> List[Dict[str, str]]:
        """Generate answers using improved keyword matching"""
        qa_results = []
        sentences = [s.strip() for s in document_text.split('.') if s.strip()]
        
        for question in questions:
            try:
                answer = self._find_answer_in_text(question, sentences)
                qa_results.append({
                    'question': question,
                    'answer': answer,
                    'method': 'keyword_matching'
                })
            except Exception as e:
                logger.warning(f"Error answering '{question}': {str(e)}")
                qa_results.append({
                    'question': question,
                    'answer': "Unable to find answer in document.",
                    'method': 'error'
                })
        
        return qa_results

    def _find_answer_in_text(self, question: str, sentences: List[str], context_window: int = 2) -> str:
        """Find answer using enhanced keyword matching"""
        question_lower = question.lower()
        
        # Remove stop words and extract meaningful keywords
        stop_words = {
            'what', 'is', 'the', 'how', 'does', 'are', 'there', 'any', 'and', 
            'or', 'for', 'in', 'on', 'at', 'to', 'a', 'an', 'can', 'will',
            'would', 'should', 'could', 'when', 'where', 'why', 'who', 'which'
        }
        
        keywords = [
            word.strip('?.,!') for word in question_lower.split() 
            if word.strip('?.,!') not in stop_words and len(word.strip('?.,!')) > 2
        ]
        
        if not keywords:
            return "Question too vague to extract meaningful keywords."
        
        # Score sentences based on keyword matches
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Keyword matching score
            keyword_score = sum(1 for keyword in keywords if keyword in sentence_lower)
            
            # Exact phrase bonus
            phrase_bonus = 2 if any(kw in sentence_lower for kw in keywords if len(kw) > 4) else 0
            
            # Length preference (not too short, not too long)
            length_score = 1 if 20 <= len(sentence) <= 200 else 0.5
            
            total_score = keyword_score + phrase_bonus + length_score
            
            if total_score > 0:
                sentence_scores.append((total_score, i, sentence))
        
        if not sentence_scores:
            return "No relevant information found for this question."
        
        # Get best match with context
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_index, best_sentence = sentence_scores[0]
        
        # Extract context window
        start_idx = max(0, best_index - context_window)
        end_idx = min(len(sentences), best_index + context_window + 1)
        
        context_sentences = sentences[start_idx:end_idx]
        answer = '. '.join(context_sentences).strip()
        
        # Limit answer length
        if len(answer) > 500:
            answer = answer[:497] + "..."
        
        return answer

    def batch_process_urls(self, urls: List[str], questions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process multiple URLs in batch"""
        results = []
        
        for url in urls:
            try:
                result = self.process_document_from_url(url, questions)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {url}: {str(e)}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'source_url': url
                })
        
        return results

    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        # Check S3 model availability
        available_models = self.model_manager.list_available_models()
        
        return {
            'status': 'healthy',
            'models_loaded': self._models_loaded,
            'model_bucket': self.model_bucket,
            'available_models': available_models,
            'gpu_available': self._has_gpu(),
            'temp_files_count': len(self._temp_files),
            'document_cache_dir': self.document_cache_dir
        }

    def cleanup(self):
        """Clean up all resources"""
        logger.info("Starting cleanup...")
        
        # Clean up temporary document files
        for temp_file in self._temp_files.copy():
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                self._temp_files.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove {temp_file}: {str(e)}")
        
        # Clean up document cache directory
        if os.path.exists(self.document_cache_dir):
            shutil.rmtree(self.document_cache_dir)
        
        # Clean up model cache
        self.model_manager.cleanup()
        
        # Clear environment variables
        model_env_vars = list(self.model_manager.model_mappings.values()) + ['MODEL_CACHE_DIR']
        for env_var in model_env_vars:
            if env_var in os.environ:
                del os.environ[env_var]
        
        logger.info("Cleanup completed")


