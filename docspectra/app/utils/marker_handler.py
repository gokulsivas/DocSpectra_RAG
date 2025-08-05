import os
import logging
from typing import Optional
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_pdf_from_url(url: str) -> Optional[str]:
    """
    Parse PDF from URL or local path and extract text.
    
    Args:
        url (str): URL to PDF file or local file path
        
    Returns:
        Optional[str]: Extracted text or None if parsing failed
    """
    try:
        # Validate input
        if not url or not isinstance(url, str):
            logger.error("Invalid URL provided")
            return None
            
        logger.info(f"Starting PDF parsing for: {url}")
        
        # Always treat as URL - ensure it has proper protocol
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            logger.info(f"Added https:// protocol to URL: {url}")
        
        logger.info(f"Processing PDF from URL: {url}")
        
        # Load OCR + layout models
        logger.info("Loading models...")
        models = create_model_dict()
        
        # Create the PDF converter
        logger.info("Creating PDF converter...")
        converter = PdfConverter(artifact_dict=models)
        
        # Convert PDF from URL (or local path)
        logger.info("Converting PDF...")
        rendered = converter(url)
        
        # Extract plain text / markdown from rendered result
        logger.info("Extracting text...")
        text, metadata, images = text_from_rendered(rendered)
        
        if text:
            logger.info(f"Successfully extracted {len(text)} characters")
            return text
        else:
            logger.warning("No text extracted from PDF")
            return None
            
    except Exception as e:
        logger.error(f"PDF parsing failed for {url}: {str(e)}")
        return None
