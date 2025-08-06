import os
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List
import boto3
from botocore.exceptions import ClientError
import threading
from functools import lru_cache
import requests
import sys
from urllib.parse import urlparse
import mimetypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3ModelCache:
    """S3 model cache with temporary local storage"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.region = region
        self.cache_dir = tempfile.mkdtemp(prefix='marker_models_')
        self._cache_lock = threading.Lock()
        self._downloaded_models = set()
        
        # Initialize S3 client with IAM role (no explicit credentials needed)
        self.s3_client = boto3.client('s3', region_name=region)
        
        logger.info(f"Initialized S3ModelCache with temp dir: {self.cache_dir}")
    
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
        except Exception as e:
            logger.error(f"Unexpected error downloading {model_name}: {str(e)}")
            raise
    
    def get_model_path(self, model_name: str) -> str:
        """Get local path for model, downloading if necessary"""
        with self._cache_lock:
            return self._download_model_from_s3(model_name)
    
    def cleanup(self):
        """Clean up temporary cache directory"""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleaned up cache directory: {self.cache_dir}")

class OptimizedMarkerHandler:
    """Enhanced marker handler using S3-stored models"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.region = region
        self.model_cache = S3ModelCache(bucket_name, region)
        
        # Model path mappings for marker
        self.model_mappings = {
            'layout_detection': 'LAYOUT_MODEL_CHECKPOINT',
            'text_detection': 'DETECTOR_MODEL_CHECKPOINT', 
            'text_recognition': 'RECOGNITION_MODEL_CHECKPOINT',
            'table_recognition': 'TABLE_REC_MODEL_CHECKPOINT',
            'texify': 'TEXIFY_MODEL_CHECKPOINT'
        }
        
        # Document cache for temporary files
        self.document_cache_dir = tempfile.mkdtemp(prefix='marker_documents_')
        
        logger.info("Initialized OptimizedMarkerHandler")
    
    def _download_document_from_url(self, url: str, timeout: int = 300) -> str:
        """Download document from remote URL to temporary local file
        
        Args:
            url: Remote URL of the document
            timeout: Download timeout in seconds (default: 5 minutes)
            
        Returns:
            Local path to downloaded document
            
        Raises:
            ValueError: If URL is invalid or file type not supported
            requests.RequestException: If download fails
        """
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {url}")
            
            logger.info(f"Downloading document from URL: {url}")
            
            # Configure session with timeout and headers
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'DocSpectra-RAG/1.0 (Document Processing Bot)',
                'Accept': 'application/pdf,application/octet-stream,*/*'
            })
            
            # Download with streaming to handle large files
            response = session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Determine file extension
            content_type = response.headers.get('content-type', '').lower()
            file_extension = '.pdf'  # Default to PDF
            
            if 'pdf' in content_type:
                file_extension = '.pdf'
            else:
                # Try to get extension from URL
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
            
            # Download file in chunks
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            logger.info(f"Downloading {total_size} bytes to {temp_file.name}")
            
            with temp_file as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Log progress for large files
                        if total_size > 0 and downloaded_size % (1024 * 1024) == 0:
                            progress = (downloaded_size / total_size) * 100
                            logger.debug(f"Download progress: {progress:.1f}%")
            
            logger.info(f"Successfully downloaded document to: {temp_file.name}")
            
            # Validate file size
            file_size = os.path.getsize(temp_file.name)
            if file_size == 0:
                os.unlink(temp_file.name)
                raise ValueError("Downloaded file is empty")
            
            # Basic PDF validation (check PDF header)
            if file_extension == '.pdf':
                with open(temp_file.name, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        logger.warning("Downloaded file may not be a valid PDF")
            
            return temp_file.name
            
        except requests.exceptions.Timeout:
            raise requests.RequestException(f"Download timeout after {timeout} seconds")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading document from {url}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading document: {str(e)}")
            raise
    
    def setup_model_environment(self, models_to_load: Optional[list] = None):
        """Setup environment variables pointing to cached model paths"""
        if models_to_load is None:
            models_to_load = list(self.model_mappings.keys())
        
        for model_name in models_to_load:
            if model_name not in self.model_mappings:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            try:
                # Download model and get local path
                local_path = self.model_cache.get_model_path(model_name)
                
                # Set environment variable for marker
                env_var = self.model_mappings[model_name]
                os.environ[env_var] = local_path
                
                logger.info(f"Set {env_var} = {local_path}")
                
            except Exception as e:
                logger.error(f"Failed to setup {model_name}: {str(e)}")
                raise
        
        # Set cache directory for surya models
        os.environ['MODEL_CACHE_DIR'] = self.model_cache.cache_dir
        logger.info(f"Set MODEL_CACHE_DIR = {self.model_cache.cache_dir}")
    
    def process_document(self, document_path: str, output_path: str = None) -> Dict[str, Any]:
        """Process document using marker with S3 models"""
        try:
            # Ensure models are setup
            self.setup_model_environment()
            
            # Import marker after environment setup
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models
            
            # Load marker models (they will use our environment variables)
            model_list = load_all_models()
            
            # Convert document
            logger.info(f"Processing document: {document_path}")
            
            full_text, images, out_meta = convert_single_pdf(
                document_path, 
                model_list
            )
            
            # Save output if path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                logger.info(f"Saved output to: {output_path}")
            
            return {
                'success': True,
                'text': full_text,
                'images': images,
                'metadata': out_meta,
                'models_used': list(self.model_mappings.keys())
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_document_from_url(self, document_url: str, questions: List[str] = None, output_path: str = None) -> Dict[str, Any]:
        """Process document from URL and optionally answer questions
        
        Args:
            document_url: Remote URL of the document to process
            questions: Optional list of questions to answer about the document
            output_path: Optional path to save processed text
            
        Returns:
            Dict containing processing results and optional answers
        """
        temp_document_path = None
        
        try:
            # Download document from URL
            temp_document_path = self._download_document_from_url(document_url)
            
            # Process the downloaded document
            processing_result = self.process_document(temp_document_path, output_path)
            
            if not processing_result['success']:
                return processing_result
            
            result = {
                'success': True,
                'text': processing_result['text'],
                'images': processing_result['images'],
                'metadata': processing_result['metadata'],
                'models_used': processing_result['models_used'],
                'source_url': document_url
            }
            
            # If questions provided, generate answers using basic text search
            if questions:
                result['answers'] = self._generate_answers(processing_result['text'], questions)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document from URL {document_url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'source_url': document_url
            }
        
        finally:
            # Cleanup temporary document file
            if temp_document_path and os.path.exists(temp_document_path):
                try:
                    os.unlink(temp_document_path)
                    logger.debug(f"Cleaned up temporary document: {temp_document_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary file {temp_document_path}: {str(e)}")
    
    def _generate_answers(self, document_text: str, questions: List[str]) -> List[str]:
        """Generate answers to questions based on document text
        
        This is a basic implementation using keyword search and context extraction.
        For production use, consider integrating with a proper Q&A model or RAG system.
        """
        answers = []
        
        # Split document into sentences for better context extraction
        sentences = [s.strip() for s in document_text.split('.') if s.strip()]
        
        for question in questions:
            try:
                answer = self._find_answer_in_text(question, sentences)
                answers.append(answer)
            except Exception as e:
                logger.warning(f"Error generating answer for question '{question}': {str(e)}")
                answers.append("Unable to find answer in the document.")
        
        return answers
    
    def _find_answer_in_text(self, question: str, sentences: List[str], context_window: int = 2) -> str:
        """Find answer to question in document sentences using keyword matching"""
        question_lower = question.lower()
        
        # Extract keywords from question (remove common words)
        stop_words = {'what', 'is', 'the', 'how', 'does', 'are', 'there', 'any', 'and', 'or', 'for', 'in', 'on', 'at', 'to', 'a', 'an'}
        keywords = [word.strip('?.,!') for word in question_lower.split() if word.strip('?.,!') not in stop_words]
        
        # Score sentences based on keyword matches
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            score = sum(1 for keyword in keywords if keyword in sentence_lower)
            if score > 0:
                sentence_scores.append((score, i, sentence))
        
        if not sentence_scores:
            return "No relevant information found in the document."
        
        # Get the best matching sentence and its context
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_index, best_sentence = sentence_scores[0]
        
        # Extract context (surrounding sentences)
        start_idx = max(0, best_index - context_window)
        end_idx = min(len(sentences), best_index + context_window + 1)
        
        context_sentences = sentences[start_idx:end_idx]
        answer = '. '.join(context_sentences).strip()
        
        # Limit answer length
        if len(answer) > 500:
            answer = answer[:497] + "..."
        
        return answer
    
    def list_available_models(self) -> Dict[str, bool]:
        """Check which models are available in S3"""
        available = {}
        
        for model_name in self.model_mappings.keys():
            try:
                # Check if model exists in S3
                response = self.model_cache.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=f"models/{model_name}/",
                    MaxKeys=1
                )
                available[model_name] = 'Contents' in response
            except Exception:
                available[model_name] = False
        
        return available
    
    def health_check(self) -> Dict[str, Any]:
        """System health check"""
        try:
            # Check S3 connectivity
            self.model_cache.s3_client.head_bucket(Bucket=self.bucket_name)
            s3_status = 'ok'
        except Exception as e:
            s3_status = f'error: {str(e)}'
        
        # Check available models
        available_models = self.list_available_models()
        
        return {
            'status': 'healthy' if s3_status == 'ok' else 'degraded',
            's3_connection': s3_status,
            'cache_directory': self.model_cache.cache_dir,
            'document_cache_directory': self.document_cache_dir,
            'available_models': available_models,
            'downloaded_models': list(self.model_cache._downloaded_models)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        # Cleanup model cache
        self.model_cache.cleanup()
        
        # Cleanup document cache
        if os.path.exists(self.document_cache_dir):
            shutil.rmtree(self.document_cache_dir)
            logger.info(f"Cleaned up document cache directory: {self.document_cache_dir}")
        
        # Clear environment variables
        for env_var in self.model_mappings.values():
            if env_var in os.environ:
                del os.environ[env_var]
        
        if 'MODEL_CACHE_DIR' in os.environ:
            del os.environ['MODEL_CACHE_DIR']
        
        logger.info("Cleanup completed")

# Context manager for automatic cleanup
class MarkerS3Context:
    """Context manager for marker S3 handler"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.region = region
        self.handler = None
    
    def __enter__(self):
        self.handler = OptimizedMarkerHandler(self.bucket_name, self.region)
        return self.handler
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler:
            self.handler.cleanup()
