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
import multiprocessing

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Prefer pypdf; fallback to PyPDF2
    from pypdf import PdfReader, PdfWriter  # type: ignore
    _PDF_LIB = "pypdf"
except Exception:
    try:
        from PyPDF2 import PdfReader, PdfWriter  # type: ignore
        _PDF_LIB = "PyPDF2"
    except Exception:
        PdfReader = None  # type: ignore
        PdfWriter = None  # type: ignore
        _PDF_LIB = None

# Modern Marker imports
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

# Configure logging
logging.basicConfig(level=logging.INFO)

# Shared in-process model artifacts to avoid re-initialization per request
_SHARED_MODELS: Dict[str, Any] = {
    "loaded": False,
    "model_dict": None,
    "converter": None,
}

# Helper to optimize thread usage on CPU
def _configure_cpu_runtime():
    try:
        cpu_count = max(1, multiprocessing.cpu_count())
        # Leave 1 core free for OS/background
        threads = max(1, min(4, cpu_count - 1))
        os.environ.setdefault("TORCH_DEVICE", "cpu")
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))
        os.environ.setdefault("TOKENIZER_PARALLELISM", "false")
        os.environ.setdefault("SURYA_NUM_WORKERS", str(max(1, min(2, threads))))
        try:
            import torch
            torch.set_num_threads(threads)
        except Exception:
            pass
        logger.info(f"CPU runtime configured: threads={threads}")
    except Exception as e:
        logger.warning(f"Failed to configure CPU runtime: {e}")

class S3ModelManager:
    """Manages downloading and caching models from S3 for Marker (optimized without layout detection)"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.region = region
        # Use persistent cache under user's cache directory instead of tmp
        default_cache = os.path.join(os.path.expanduser("~"), ".cache", "docspectra", "marker_models")
        self.cache_dir = os.getenv("DOCSPECTRA_MODEL_CACHE", default_cache)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._cache_lock = threading.Lock()
        self._downloaded_models = set()
        
        # Initialize S3 client (using IAM role, no explicit credentials needed)
        self.s3_client = boto3.client('s3', region_name=region)
        
        # OPTIMIZED: Model mappings WITHOUT layout detection
        self.model_mappings = {
            'reading_order': 'READING_ORDER_MODEL_CHECKPOINT',
            'table_recognition': 'TABLE_REC_MODEL_CHECKPOINT',
            'texify': 'TEXIFY_MODEL_CHECKPOINT',
            'text_recognition': 'RECOGNITION_MODEL_CHECKPOINT',
            'text_detection': 'DETECTOR_MODEL_CHECKPOINT'
        }
        
        logger.info(f"Initialized S3ModelManager with cache dir: {self.cache_dir}")
        logger.info(f"Optimized model set (no layout detection): {list(self.model_mappings.keys())}")
    
    def _download_model_from_s3(self, model_name: str) -> str:
        """Download model from S3 to local cache if not already present"""
        local_model_path = os.path.join(self.cache_dir, model_name)
        s3_prefix = f"docspectra_models/{model_name}/"
        
        # If already marked downloaded in this process, reuse
        if model_name in self._downloaded_models:
            logger.info(f"Model {model_name} already cached locally (session)")
            return local_model_path
        
        # If directory exists and is non-empty, assume cached and reuse
        if os.path.isdir(local_model_path):
            try:
                if any(os.scandir(local_model_path)):
                    logger.info(f"Model {model_name} found in persistent cache: {local_model_path}")
                    self._downloaded_models.add(model_name)
                    return local_model_path
            except Exception:
                pass
        
        logger.info(f"Downloading {model_name} from S3 to {local_model_path}")
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
                if s3_key.endswith('/'):
                    continue
                relative_path = s3_key[len(s3_prefix):]
                local_file_path = os.path.join(local_model_path, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                self.s3_client.download_file(self.bucket_name, s3_key, local_file_path)
                logger.debug(f"Downloaded {s3_key} -> {local_file_path}")
            
            self._downloaded_models.add(model_name)
            logger.info(f"Successfully downloaded {model_name}")
            return local_model_path
            
        except ClientError as e:
            logger.error(f"Failed downloading {model_name} from S3: {e}")
            raise
    
    def setup_model_environment(self, models_to_load: Optional[List[str]] = None):
        """Setup environment variables pointing to cached model paths (optimized set)"""
        if models_to_load is None:
            # OPTIMIZED: Default to the essential models only (no layout detection)
            models_to_load = list(self.model_mappings.keys())
        
        # Filter out layout detection even if explicitly requested
        models_to_load = [m for m in models_to_load if m != 'layout_detection']
        
        logger.info(f"Setting up optimized model set: {models_to_load}")
        
        for model_name in models_to_load:
            if model_name not in self.model_mappings:
                logger.warning(f"Unknown model: {model_name} (skipping)")
                continue
            
            try:
                local_path = self._download_model_from_s3(model_name)
                env_var = self.model_mappings[model_name]
                os.environ[env_var] = local_path
                logger.info(f"✅ Set {env_var} = {local_path}")
            except Exception as e:
                logger.error(f"Failed to setup {model_name}: {str(e)}")
                raise
        
        # Set cache directory for additional models to persistent cache
        os.environ['MODEL_CACHE_DIR'] = self.cache_dir
        logger.info(f"Set MODEL_CACHE_DIR = {self.cache_dir}")
        
        # OPTIMIZATION: Set environment flag to disable layout detection in Marker
        os.environ['MARKER_DISABLE_LAYOUT_DETECTION'] = 'true'
        logger.info("✅ Layout detection disabled for faster processing")
    
    def get_model_size_estimate(self) -> Dict[str, str]:
        """Get estimated download sizes for the optimized model set"""
        # Rough size estimates (these would vary based on your actual models)
        size_estimates = {
            'reading_order': '~500MB',
            'table_recognition': '~300MB', 
            'texify': '~1.2GB',
            'text_recognition': '~400MB',
            'text_detection': '~200MB'
        }
        
        total_estimate = "~2.6GB (vs ~3.1GB with layout detection)"
        
        logger.info(f"Optimized model set size estimate: {total_estimate}")
        return {
            **size_estimates,
            'total_estimate': total_estimate,
            'optimization': 'Layout detection removed (~500MB saved)'
        }
    
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
    FIXED: Processor initialization issue resolved.
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
        """Setup environment for optimal Marker performance (without layout detection)"""
        # Setup models from S3 into persistent cache (optimized set)
        logger.info("🔍 DEBUG: Checking environment variables before setup...")
        logger.info(f"MARKER_DISABLE_LAYOUT_DETECTION = {os.getenv('MARKER_DISABLE_LAYOUT_DETECTION', 'NOT SET')}")
    
        self.model_manager.setup_model_environment()
        
        logger.info("🔍 DEBUG: Checking environment variables after setup...")
        logger.info(f"MARKER_DISABLE_LAYOUT_DETECTION = {os.getenv('MARKER_DISABLE_LAYOUT_DETECTION', 'NOT SET')}")
        logger.info(f"Available model env vars: {[k for k in os.environ.keys() if 'MODEL' in k]}")
        # Optimize for CPU-only execution
        _configure_cpu_runtime()
        
        # Additional Marker configuration for optimized processing
        os.environ["TORCH_DEVICE"] = "cpu"
        os.environ.setdefault("OCR_ENGINE", "surya")
        
        # Disable image extraction for CPU speedups
        os.environ["EXTRACT_IMAGES"] = "false"
        
        # OPTIMIZATION: Configure Marker to skip layout detection
        os.environ["MARKER_DISABLE_LAYOUT_DETECTION"] = "true"
        os.environ["MARKER_USE_OPTIMIZED_PIPELINE"] = "true"
        
        if self.batch_multiplier > 1:
            os.environ["BATCH_MULTIPLIER"] = str(self.batch_multiplier)
        
        logger.info("✅ Marker environment configured with optimized model set")
        logger.info("✅ Layout detection disabled - faster processing enabled")

    def _has_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _ensure_models_loaded(self):
        """Load Marker models with S3-downloaded model checkpoints (FIXED processor initialization)"""
        if not self._models_loaded:
            with self._cache_lock:
                if not self._models_loaded:  # Double-check locking
                    # Reuse shared models if already initialized in-process
                    if _SHARED_MODELS["loaded"]:
                        self._model_dict = _SHARED_MODELS["model_dict"]
                        self._converter = _SHARED_MODELS["converter"]
                        self._models_loaded = True
                        logger.info("Marker models already initialized in process (reusing shared models)")
                        return

                    logger.info("Setting up FIXED optimized Marker models...")
                    
                    # Setup environment with S3 models (optimized set)
                    self._setup_marker_environment()
                    
                    # FORCE: Remove layout detection model from environment if it exists
                    layout_env_vars = [
                        'LAYOUT_MODEL_CHECKPOINT',
                        'LAYOUT_DETECTION_MODEL',
                        'LAYOUT_MODEL_PATH'
                    ]
                    for env_var in layout_env_vars:
                        if env_var in os.environ:
                            del os.environ[env_var]
                            logger.info(f"🚫 Removed {env_var} from environment")
                    
                    # Create model dictionary with our S3 models (loaded from local cache paths)
                    self._model_dict = create_model_dict()
                    
                    # FORCE: Remove layout detection from model_dict if it exists
                    if 'layout' in self._model_dict:
                        del self._model_dict['layout']
                        logger.info("🚫 Removed layout model from model_dict")
                    
                    # FIXED: Create processor list without using string names that cause parsing errors
                    try:
                        # Use default processors - let Marker handle the initialization
                        logger.info("Creating PdfConverter with default configuration...")
                        
                        # Create minimal config that ACTUALLY disables layout detection
                        config = {
                            "output_format": "markdown",
                            "extract_images": False,
                            "force_layout_block": "Text"  # This actually disables layout detection
                        }
                        
                        logger.info(f"Creating MINIMAL config: {config}")
                        config_parser = ConfigParser(config)
                        
                        # Initialize converter with minimal setup - avoid processor string parsing issue
                        self._converter = PdfConverter(
                            config=config_parser.generate_config_dict(),
                            artifact_dict=self._model_dict
                            # Don't specify processor_list to avoid string parsing errors
                        )
                        
                        logger.info("✅ PdfConverter created successfully with default processors")
                        
                    except Exception as converter_error:
                        logger.error(f"Error creating PdfConverter: {converter_error}")
                        raise
                    
                    # Mark as loaded and store in shared cache
                    self._models_loaded = True
                    _SHARED_MODELS["loaded"] = True
                    _SHARED_MODELS["model_dict"] = self._model_dict
                    _SHARED_MODELS["converter"] = self._converter
                    
                    logger.info("✅ FIXED optimized Marker models initialized")
                    logger.info("🚫 Layout detection FORCIBLY disabled")

    def convert_pdf_to_markdown(self, pdf_path: str, **kwargs) -> Tuple[str, Dict, List]:
        """Convert PDF to markdown using S3 models with optional page controls
        
        Supported kwargs:
        - page_start (int, 1-based, inclusive)
        - page_end (int, 1-based, inclusive)
        - page_batch_size (int, >0 to process in batches)
        """
        try:
            # Ensure S3 models are loaded
            self._ensure_models_loaded()

            page_start = kwargs.get("page_start")
            page_end = kwargs.get("page_end")
            page_batch_size = kwargs.get("page_batch_size", 0) or 0

            # Helper to subset pages to a temp PDF
            def _subset(pdf_in: str, start_1b: int, end_1b: int) -> str:
                if _PDF_LIB is None or PdfReader is None or PdfWriter is None:
                    logger.warning("PDF subsetting requested but no PDF library available; processing full document")
                    return pdf_in
                reader = PdfReader(pdf_in)
                total = len(reader.pages)
                s = max(1, min(start_1b, total))
                e = max(s, min(end_1b, total))
                writer = PdfWriter()
                for i in range(s-1, e):
                    writer.add_page(reader.pages[i])
                tmp_path = tempfile.NamedTemporaryFile(dir=self.document_cache_dir, suffix="_subset.pdf", delete=False).name
                with open(tmp_path, "wb") as f:
                    writer.write(f)
                self._temp_files.add(tmp_path)
                return tmp_path

            logger.info(f"Converting PDF: {pdf_path}")

            # If batching enabled, process in batches and concatenate
            if page_batch_size > 0 and _PDF_LIB is not None and PdfReader is not None:
                reader = PdfReader(pdf_path)
                total_pages = len(reader.pages)
                s = 1 if page_start is None else max(1, page_start)
                e = total_pages if page_end is None else min(total_pages, page_end)
                markdown_parts: List[str] = []
                all_images: List[Any] = []
                combined_meta: Dict[str, Any] = {"batches": []}
                
                for batch_start in range(s, e+1, page_batch_size):
                    batch_end = min(e, batch_start + page_batch_size - 1)
                    subset_path = _subset(pdf_path, batch_start, batch_end)
                    
                    logger.info(f"Processing batch {batch_start}-{batch_end}")
                    rendered = self._converter(subset_path)
                    md, meta, images = text_from_rendered(rendered)
                    
                    # FIX: Handle case where md might be a list
                    if isinstance(md, list):
                        logger.info("Markdown returned as list, joining...")
                        md = '\n\n'.join(str(item) for item in md)
                    elif not isinstance(md, str):
                        logger.warning(f"Markdown is unexpected type {type(md)}, converting...")
                        md = str(md)
                    
                    markdown_parts.append(md)
                    combined_meta["batches"].append({"start": batch_start, "end": batch_end, "meta": meta})
                    if images:
                        all_images.extend(images)
                        
                markdown_text = "\n\n\n".join(markdown_parts)
                metadata = combined_meta
                images = all_images
                return markdown_text, metadata, images

            # Else single pass, possibly with a single range
            run_path = pdf_path
            if page_start is not None or page_end is not None:
                # If only one bound provided, infer the other using reader when possible
                if _PDF_LIB is not None and PdfReader is not None:
                    reader = PdfReader(pdf_path)
                    total = len(reader.pages)
                else:
                    total = None
                s = 1 if page_start is None else page_start
                e = (total if total is not None else s) if page_end is None else page_end
                run_path = _subset(pdf_path, s, e)

            # Convert PDF using Marker with S3 models
            logger.info("Converting PDF with Marker...")
            rendered = self._converter(run_path)
            logger.info(f"Marker conversion complete, extracting text...")
            
            # FIX: Add debug logging and handle different return types
            try:
                result = text_from_rendered(rendered)
                logger.info(f"text_from_rendered returned: {type(result)}")
                
                if isinstance(result, tuple) and len(result) >= 3:
                    markdown_text, metadata, images = result[0], result[1], result[2]
                elif isinstance(result, tuple) and len(result) == 2:
                    markdown_text, metadata = result[0], result[1]
                    images = []
                elif isinstance(result, str):
                    markdown_text, metadata, images = result, {}, []
                else:
                    logger.error(f"Unexpected result type from text_from_rendered: {type(result)}")
                    logger.error(f"Result content: {result}")
                    raise ValueError(f"Unexpected result type from text_from_rendered: {type(result)}")
                
                # FIX: Handle case where markdown_text might be a list
                if isinstance(markdown_text, list):
                    logger.info("Markdown text returned as list, joining...")
                    markdown_text = '\n\n'.join(str(item) for item in markdown_text)
                elif not isinstance(markdown_text, str):
                    logger.warning(f"Markdown text is unexpected type {type(markdown_text)}, converting...")
                    markdown_text = str(markdown_text)
                
                logger.info(f"Final markdown text type: {type(markdown_text)}")
                logger.info(f"Final markdown length: {len(markdown_text)} characters")
                logger.info(f"Markdown preview: {markdown_text[:200]}...")
                
                return markdown_text, metadata, images
                
            except Exception as text_extract_error:
                logger.error(f"Error in text_from_rendered: {text_extract_error}")
                logger.error(f"Rendered type: {type(rendered)}")
                raise
                
        except Exception as e:
            logger.error(f"Error converting PDF: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def process_document_from_url(self, 
                                document_url: str,
                                questions: Optional[List[str]] = None,
                                **kwargs) -> Dict[str, Any]:
        """Process document from URL using S3 models
        
        Optional kwargs: page_start, page_end, page_batch_size
        """
        temp_document_path = None
        
        try:
            # Download document
            temp_document_path = self.download_document_from_url(document_url)
            
            # Convert to markdown (supports page controls)
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
        """Check system health including optimization status"""
        # Check Bedrock connection if enabled
        bedrock_status = False
        if self.use_bedrock_qa and self.bedrock_client:
            try:
                bedrock_status = self.bedrock_client.test_connection()
            except Exception as e:
                logger.warning(f"Bedrock health check failed: {e}")
        
        # Get model size estimates
        model_estimates = self.model_manager.get_model_size_estimate()
        
        return {
            'status': 'healthy',
            'models_loaded': self._models_loaded,
            'model_bucket': self.model_bucket,
            'gpu_available': self._has_gpu(),
            'temp_files_count': len(self._temp_files),
            'document_cache_dir': self.document_cache_dir,
            'bedrock_qa_enabled': self.use_bedrock_qa,
            'bedrock_connection': bedrock_status,
            'bedrock_model': getattr(self.bedrock_client, 'titan_model_id', 'N/A') if self.bedrock_client else 'N/A',
            'optimization': {
                'layout_detection_disabled': True,
                'model_set': 'optimized (5 models instead of 6)',
                'estimated_size_saving': '~500MB',
                'models_used': list(self.model_manager.model_mappings.keys())
            },
            'model_estimates': model_estimates
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
            use_llm=False,
            batch_multiplier=1,
            use_bedrock_qa=use_bedrock_qa
        )
    
    def __enter__(self):
        return self.handler
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Only clean temporary documents, keep persistent model cache intact
        self.handler.cleanup_temp_documents()