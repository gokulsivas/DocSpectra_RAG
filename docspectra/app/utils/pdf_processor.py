# app/utils/pdf_processor.py - PDF processing with PyPDF + Tesseract
import logging
import tempfile
import os
from typing import Dict, Any, Optional
from pathlib import Path
import requests
from urllib.parse import urlparse

import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

from ..config import ocr_config

logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF processor using PyPDF and Tesseract OCR"""
    
    def __init__(self):
        self.tesseract_cmd = ocr_config.tesseract_cmd
        self.ocr_lang = ocr_config.ocr_lang
        self.dpi = ocr_config.dpi
        self.timeout = ocr_config.timeout
        
        # Set tesseract command path if specified
        if self.tesseract_cmd != 'tesseract':
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        
        logger.info(f"PDFProcessor initialized with Tesseract: {self.tesseract_cmd}")
        logger.info(f"OCR language: {self.ocr_lang}, DPI: {self.dpi}")
    
    def download_pdf_from_url(self, url: str) -> str:
        """Download PDF from URL to temporary file"""
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {url}")
            
            logger.info(f"Downloading PDF from: {url}")
            
            # Configure session
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'DocSpectra/1.0 (PDF Processing)',
                'Accept': 'application/pdf,application/octet-stream,*/*'
            })
            
            # Download with streaming
            response = session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            
            # Download in chunks
            with temp_file as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Validate file size
            file_size = os.path.getsize(temp_file.name)
            if file_size == 0:
                os.unlink(temp_file.name)
                raise ValueError("Downloaded file is empty")
            
            # Basic PDF validation
            with open(temp_file.name, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    os.unlink(temp_file.name)
                    raise ValueError("Downloaded file is not a valid PDF")
            
            logger.info(f"Successfully downloaded PDF: {file_size} bytes")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from PDF using PyPDF first, then OCR if needed"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Try text extraction with PyPDF first
            text_content = self._extract_text_pypdf(pdf_path)
            
            # Check if we got meaningful text
            if self._has_meaningful_text(text_content):
                logger.info(f"Extracted text using PyPDF: {len(text_content)} characters")
                return {
                    'success': True,
                    'text': text_content,
                    'method': 'pypdf',
                    'pages_processed': self._count_pdf_pages(pdf_path)
                }
            
            # If PyPDF didn't work well, try OCR
            logger.info("PyPDF extraction insufficient, trying OCR...")
            ocr_content = self._extract_text_ocr(pdf_path)
            
            if ocr_content and len(ocr_content.strip()) > 50:
                logger.info(f"Extracted text using OCR: {len(ocr_content)} characters")
                return {
                    'success': True,
                    'text': ocr_content,
                    'method': 'ocr',
                    'pages_processed': self._count_pdf_pages(pdf_path)
                }
            
            # If both methods failed
            return {
                'success': False,
                'error': 'Could not extract meaningful text from PDF',
                'text': text_content or ocr_content or '',
                'method': 'failed'
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'method': 'error'
            }
    
    def _extract_text_pypdf(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            text_content = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            text_content.append(text.strip())
                            logger.debug(f"Extracted text from page {page_num + 1}: {len(text)} chars")
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
            
            return '\n\n'.join(text_content)
            
        except Exception as e:
            logger.error(f"PyPDF extraction error: {e}")
            return ''
    
    def _extract_text_ocr(self, pdf_path: str) -> str:
        """Extract text using Tesseract OCR"""
        try:
            logger.info("Converting PDF to images for OCR...")
            
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=1,
                last_page=10,  # Limit to first 10 pages for performance
                poppler_path=None  # Use system poppler
            )
            
            if not images:
                logger.warning("No images converted from PDF")
                return ''
            
            logger.info(f"Converted PDF to {len(images)} images")
            
            # OCR each image
            text_content = []
            for i, image in enumerate(images):
                try:
                    # Preprocess image for better OCR
                    processed_image = self._preprocess_image(image)
                    
                    # OCR the image
                    text = pytesseract.image_to_string(
                        processed_image,
                        lang=self.ocr_lang,
                        timeout=self.timeout,
                        config='--psm 6'  # Uniform block of text
                    )
                    
                    if text and text.strip():
                        text_content.append(text.strip())
                        logger.debug(f"OCR page {i + 1}: {len(text)} chars")
                
                except Exception as e:
                    logger.warning(f"Error OCR processing page {i + 1}: {e}")
                    continue
            
            return '\n\n'.join(text_content)
            
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return ''
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Simple thresholding to improve contrast
            threshold = 128
            img_array = np.where(img_array > threshold, 255, 0)
            
            # Convert back to PIL Image
            return Image.fromarray(img_array.astype('uint8'))
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def _has_meaningful_text(self, text: str) -> bool:
        """Check if extracted text is meaningful"""
        if not text or len(text.strip()) < 50:
            return False
        
        # Check for reasonable word count
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check for reasonable character distribution
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars / len(text) < 0.3:  # Less than 30% alphabetic characters
            return False
        
        return True
    
    def _count_pdf_pages(self, pdf_path: str) -> int:
        """Count pages in PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return len(pdf_reader.pages)
        except Exception:
            return 0
    
    def process_document_from_url(self, url: str) -> Dict[str, Any]:
        """Complete pipeline: download and extract text from PDF URL"""
        temp_path = None
        try:
            # Download PDF
            temp_path = self.download_pdf_from_url(url)
            
            # Extract text
            result = self.extract_text_from_pdf(temp_path)
            result['source_url'] = url
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document from URL {url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'source_url': url,
                'text': '',
                'method': 'error'
            }
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_path}: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check if OCR system is working"""
        try:
            # Test Tesseract
            test_image = Image.new('RGB', (100, 50), color='white')
            test_text = pytesseract.image_to_string(test_image)
            
            return {
                'status': 'healthy',
                'tesseract_cmd': self.tesseract_cmd,
                'ocr_lang': self.ocr_lang,
                'tesseract_version': pytesseract.get_tesseract_version(),
                'test_successful': True
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'tesseract_cmd': self.tesseract_cmd,
                'test_successful': False
            }