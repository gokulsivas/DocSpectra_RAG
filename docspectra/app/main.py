# app/main.py - Enhanced version
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .router import router
from dotenv import load_dotenv
import logging
import os

# Load environment first
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="DocSpectra - HackRx 6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(router, prefix="/hackrx")

@app.on_startup
async def startup_event():
    """Validate environment and pre-warm services"""
    logger.info("🚀 Starting DocSpectra RAG API...")
    
    # Validate required environment variables
    required_vars = ['PINECONE_API_KEY', 'AWS_DEFAULT_REGION', 'MARKER_S3_BUCKET']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"❌ Missing environment variables: {missing}")
        raise Exception(f"Missing: {missing}")
    
    logger.info("✅ All services ready!")

@app.on_shutdown
async def shutdown_event():
    """Cleanup resources"""
    logger.info("👋 Shutting down gracefully...")
