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
        self.cache_dir = tempfile.mkdtemp(prefix='docspectra_models_')
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
        s3_prefix = f"docspectra_models/{model_name}/"
        
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
    

    
    def cleanup(self):
        """Clean up temporary cache directory"""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleaned up model cache directory: {self.cache_dir}")

class BedrockClient:
    """Bedrock client for Titan model integration"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self._bedrock_client = None
        
        # Titan model configurations
        self.titan_model_id = os.getenv('TITAN_MODEL_ID', 'amazon.titan-text-express-v1')
        self.max_tokens = int(os.getenv('MAX_TOKENS', '300'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        self.top_p = float(os.getenv('TOP_P', '0.9'))
        
        logger.info(f"Initialized BedrockClient with model: {self.titan_model_id}")
    
    @property
    def bedrock_client(self):
        """Lazy-loaded Bedrock runtime client"""
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=self.region
            )
            logger.info("Bedrock runtime client initialized")
        return self._bedrock_client
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Bedrock Titan model"""
        try:
            # Build enhanced prompt
            prompt = self._build_prompt(question, context)
            
            # Prepare request body
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "temperature": self.temperature,
                    "maxTokenCount": self.max_tokens,
                    "topP": self.top_p,
                    "stopSequences": ["Human:", "Context:", "Question:", "\n\n---"]
                }
            }
            
            # Invoke Bedrock model
            response = self.bedrock_client.invoke_model(
                modelId=self.titan_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body).encode("utf-8")
            )
            
            # Parse response
            result = json.loads(response['body'].read().decode())
            
            if "results" not in result or len(result["results"]) == 0:
                logger.warning("Invalid response format from Bedrock")
                return "Unable to generate answer due to invalid model response."
            
            answer = result["results"][0]["outputText"].strip()
            
            # Clean up the answer
            answer = self._clean_answer(answer)
            
            logger.info(f"Generated answer using Bedrock Titan: {len(answer)} characters")
            return answer
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Bedrock API error [{error_code}]: {error_message}")
            
            if error_code == 'ValidationException':
                return f"Invalid request to language model: {error_message}"
            elif error_code == 'ResourceNotFoundException':
                return f"Language model {self.titan_model_id} not available in region {self.region}"
            elif error_code == 'AccessDeniedException':
                return "Access denied to language model. Please check your AWS permissions."
            elif error_code == 'ThrottlingException':
                return "Language model is currently busy. Please try again later."
            else:
                return f"Language model error: {error_message}"
                
        except Exception as e:
            logger.error(f"Unexpected error generating answer: {e}")
            return f"Unable to generate answer due to technical issue: {str(e)}"
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build enhanced prompt for better answer generation"""
        return f"""You are an AI assistant that answers questions based on provided document context. 

Instructions:
- Answer the question directly and concisely based ONLY on the provided context
- If the answer is not in the context, clearly state that the information is not available
- Be accurate and avoid speculation or information not in the context
- Keep your answer focused and relevant to the question
- Use clear, professional language

Context:
{context}

Question: {question}

Answer:"""
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and format the generated answer"""
        # Remove common artifacts
        answer = answer.replace("Answer:", "").strip()
        answer = answer.replace("Based on the context", "").strip()
        answer = answer.replace("According to the document", "").strip()
        
        # Remove incomplete sentences at the end
        sentences = answer.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            answer = '.'.join(sentences[:-1]) + '.'
        
        # Ensure proper capitalization
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        # Limit length
        if len(answer) > 800:
            answer = answer[:797] + "..."
        
        return answer
    
    def test_connection(self) -> bool:
        """Test Bedrock connection"""
        try:
            test_question = "What is this?"
            test_context = "This is a test document."
            result = self.generate_answer(test_question, test_context)
            logger.info(f"Bedrock connection test successful: {result[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Bedrock connection test failed: {e}")
            return False

class MarkerHandlerWithS3Models:
    """
    Marker handler that loads models from S3 and processes documents from URLs only.
    Enhanced with Bedrock Titan integration for intelligent answer generation.
    """
    
    def __init__(self, 
                 model_bucket: str,
                 aws_region: str = "us-east-1",
                 use_llm: bool = False,
                 batch_multiplier: int = 1,
                 download_timeout: int = 300,
                 use_bedrock_qa: bool = True):
        """
        Initialize handler with S3 models and Bedrock integration.
        
        Args:
            model_bucket: S3 bucket containing Marker models (NOT documents)
            aws_region: AWS region
            use_llm: Whether to use LLM for enhanced accuracy
            batch_multiplier: Batch size multiplier for processing
            download_timeout: URL download timeout in seconds
            use_bedrock_qa: Whether to use Bedrock for Q&A (vs keyword matching)
        """
        self.model_bucket = model_bucket
        self.aws_region = aws_region
        self.use_llm = use_llm
        self.batch_multiplier = batch_multiplier
        self.download_timeout = download_timeout
        self.use_bedrock_qa = use_bedrock_qa
        
        # Initialize S3 model manager
        self.model_manager = S3ModelManager(model_bucket, aws_region)
        
        # Initialize Bedrock client if Q&A is enabled
        self.bedrock_client = None
        if self.use_bedrock_qa:
            try:
                self.bedrock_client = BedrockClient(aws_region)
                logger.info("Bedrock Q&A integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Bedrock client: {e}")
                logger.warning("Falling back to keyword-based Q&A")
                self.use_bedrock_qa = False
        
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
        logger.info(f"Bedrock Q&A: {'Enabled' if self.use_bedrock_qa else 'Disabled'}")
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
        """Generate answers using Bedrock integration or keyword matching fallback"""
        qa_results = []
        sentences = [s.strip() for s in document_text.split('.') if s.strip()]
        
        for question in questions:
            try:
                answer = self._find_answer_in_text(question, sentences, document_text)
                qa_results.append({
                    'question': question,
                    'answer': answer,
                    'method': 'bedrock_titan' if self.use_bedrock_qa else 'keyword_matching'
                })
            except Exception as e:
                logger.warning(f"Error answering '{question}': {str(e)}")
                qa_results.append({
                    'question': question,
                    'answer': "Unable to find answer in document.",
                    'method': 'error'
                })
        
        return qa_results

    def _find_answer_in_text(self, question: str, sentences: List[str], full_document_text: str, context_window: int = 3) -> str:
        """
        Enhanced answer finding with Bedrock Titan integration.
        
        First finds relevant context using keyword matching, then uses Bedrock Titan 
        to generate a proper answer based on that context.
        """
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
        
        logger.info(f"Extracted keywords for '{question}': {keywords}")
        
        # Score sentences based on keyword matches
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Keyword matching score
            keyword_score = sum(1 for keyword in keywords if keyword in sentence_lower)
            
            # Exact phrase bonus
            phrase_bonus = 2 if any(kw in sentence_lower for kw in keywords if len(kw) > 4) else 0
            
            # Length preference (not too short, not too long)
            length_score = 1 if 20 <= len(sentence) <= 300 else 0.5
            
            total_score = keyword_score + phrase_bonus + length_score
            
            if total_score > 0:
                sentence_scores.append((total_score, i, sentence))
        
        if not sentence_scores:
            return "No relevant information found for this question."
        
        # Get best matches and create context
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Take top scoring sentences to build context
        top_sentences = sentence_scores[:5]  # Top 5 relevant sentences
        context_parts = []
        
        for score, sentence_idx, sentence in top_sentences:
            # Add context window around each relevant sentence
            start_idx = max(0, sentence_idx - context_window)
            end_idx = min(len(sentences), sentence_idx + context_window + 1)
            
            context_sentences = sentences[start_idx:end_idx]
            context_part = '. '.join(context_sentences).strip()
            
            if context_part not in context_parts:  # Avoid duplicates
                context_parts.append(context_part)
        
        # Combine all context parts
        combined_context = '\n\n---\n\n'.join(context_parts)
        
        # Limit context length to avoid token limits
        max_context_length = 4000  # Adjust based on your model's limits
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length] + "\n\n[Context truncated...]"
        
        logger.info(f"Built context of {len(combined_context)} characters from {len(context_parts)} parts")
        
        # Use Bedrock Titan for answer generation if available
        if self.use_bedrock_qa and self.bedrock_client:
            try:
                answer = self.bedrock_client.generate_answer(question, combined_context)
                logger.info("Generated answer using Bedrock Titan")
                return answer
            except Exception as e:
                logger.warning(f"Bedrock answer generation failed: {e}")
                logger.info("Falling back to keyword-based answer")
                # Fall through to keyword-based approach
        
        # Fallback: keyword-based answer (original logic)
        logger.info("Using keyword-based answer generation")
        best_score, best_index, best_sentence = sentence_scores[0]
        
        # Extract context window around best match
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
        """Check system health including Bedrock integration"""
        # Check Bedrock connection if enabled
        bedrock_status = False
        if self.use_bedrock_qa and self.bedrock_client:
            try:
                bedrock_status = self.bedrock_client.test_connection()
            except Exception as e:
                logger.warning(f"Bedrock health check failed: {e}")
        
        return {
            'status': 'healthy',
            'models_loaded': self._models_loaded,
            'model_bucket': self.model_bucket,
            'gpu_available': self._has_gpu(),
            'temp_files_count': len(self._temp_files),
            'document_cache_dir': self.document_cache_dir,
            'bedrock_qa_enabled': self.use_bedrock_qa,
            'bedrock_connection': bedrock_status,
            'bedrock_model': getattr(self.bedrock_client, 'titan_model_id', 'N/A') if self.bedrock_client else 'N/A'
        }

    def cleanup_temp_documents(self):
        """Clean up only temporary document files and document cache directory"""
        logger.info("Cleaning up temporary document files...")
        for temp_file in self._temp_files.copy():
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                self._temp_files.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove {temp_file}: {str(e)}")
        if os.path.exists(self.document_cache_dir):
            shutil.rmtree(self.document_cache_dir)
        logger.info("Temporary document cleanup completed")

    def cleanup(self):
        """Clean up all resources (manual call only)"""
        logger.info("Starting full cleanup...")
        self.cleanup_temp_documents()
        # Clean up model cache (manual only)
        self.model_manager.cleanup()
        # Clear environment variables
        model_env_vars = list(self.model_manager.model_mappings.values()) + ['MODEL_CACHE_DIR']
        for env_var in model_env_vars:
            if env_var in os.environ:
                del os.environ[env_var]
        logger.info("Cleanup completed")

# Context manager for easy usage
class MarkerS3Context:
    """Context manager for Marker with S3 models and Bedrock integration"""
    
    def __init__(self, bucket_name: str, use_bedrock_qa: bool = True):
        self.handler = MarkerHandlerWithS3Models(
            model_bucket=bucket_name,
            aws_region=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
            use_llm=False,  # Set to True for enhanced PDF processing accuracy
            batch_multiplier=1,
            use_bedrock_qa=use_bedrock_qa
        )
    
    def __enter__(self):
        return self.handler
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handler.cleanup_temp_documents()  # Only clean up temp documents after each query
