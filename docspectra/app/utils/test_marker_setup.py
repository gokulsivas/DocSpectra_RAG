# test_marker_setup.py
"""
Test script to verify Marker setup is working correctly
"""

import sys
import os
import logging
from pathlib import Path

# Add app directory to Python path
sys.path.append(str(Path(__file__).parent / 'app'))

from utils.marker_handler import MarkerHandler
from utils.s3_model_manager import S3ModelManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_s3_connection():
    """Test S3 connection and bucket access"""
    logger.info("Testing S3 connection...")
    try:
        manager = S3ModelManager()
        response = manager.s3_client.list_objects_v2(
            Bucket='markerbucket69',
            MaxKeys=5
        )
        logger.info(f"✅ S3 connection successful. Objects found: {response.get('KeyCount', 0)}")
        return True
    except Exception as e:
        logger.error(f"❌ S3 connection failed: {e}")
        return False

def test_model_availability():
    """Test if models are available locally"""
    logger.info("Testing local model availability...")
    manager = S3ModelManager()
    models = manager.check_models_exist_locally()
    
    for model_name, available in models.items():
        status = "✅" if available else "❌"
        logger.info(f"{status} {model_name} models: {'Available' if available else 'Not found'}")
    
    return all(models.values())

def test_marker_initialization():
    """Test Marker handler initialization"""
    logger.info("Testing Marker handler initialization...")
    try:
        handler = MarkerHandler(use_llm=False)  # Start without LLM for testing
        health = handler.health_check()
        
        logger.info(f"Handler status: {health['status']}")
        logger.info(f"Converter initialized: {health['converter_initialized']}")
        
        return health['status'] == 'healthy'
    except Exception as e:
        logger.error(f"❌ Marker initialization failed: {e}")
        return False

def test_pdf_conversion():
    """Test PDF conversion with temp.pdf if it exists"""
    logger.info("Testing PDF conversion...")
    
    # Check if temp.pdf exists
    test_pdf = Path("temp.pdf")
    if not test_pdf.exists():
        logger.warning("⚠️  temp.pdf not found, skipping conversion test")
        return True
    
    try:
        handler = MarkerHandler(use_llm=False)
        result = handler.convert_pdf(str(test_pdf))
        
        logger.info(f"✅ PDF conversion successful!")
        logger.info(f"   Text length: {len(result['text'])} characters")
        logger.info(f"   Images found: {len(result['images'])} images")
        logger.info(f"   Source: {result['source_file']}")
        
        # Show first 200 characters of converted text
        preview = result['text'][:200].replace('\n', ' ')
        logger.info(f"   Preview: {preview}...")
        
        return True
    except Exception as e:
        logger.error(f"❌ PDF conversion failed: {e}")
        return False

def test_document_structure():
    """Test document structure analysis"""
    logger.info("Testing document structure analysis...")
    
    test_pdf = Path("temp.pdf")
    if not test_pdf.exists():
        logger.warning("⚠️  temp.pdf not found, skipping structure test")
        return True
    
    try:
        handler = MarkerHandler()
        structure = handler.get_document_structure(str(test_pdf))
        
        logger.info("✅ Document structure analysis successful!")
        for key, value in structure.items():
            logger.info(f"   {key}: {value}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Document structure analysis failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Starting Marker setup tests...")
    
    tests = [
        ("S3 Connection", test_s3_connection),
        ("Model Availability", test_model_availability),
        ("Marker Initialization", test_marker_initialization),
        ("PDF Conversion", test_pdf_conversion),
        ("Document Structure", test_document_structure),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your Marker setup is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Please check the logs above.")
        
        # Provide troubleshooting tips
        logger.info("\n🔧 Troubleshooting tips:")
        logger.info("1. Make sure AWS CLI is configured: aws configure")
        logger.info("2. Check IAM permissions for S3 bucket access")
        logger.info("3. Ensure internet connection for model downloads")
        logger.info("4. Check disk space in ~/.cache directory")
        logger.info("5. For GPU issues, verify CUDA installation")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
