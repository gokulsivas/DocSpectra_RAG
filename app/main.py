# app/main.py - Enhanced version with modern lifespan approach

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application lifespan events.
    Everything before yield runs on startup.
    Everything after yield runs on shutdown.
    """
    # Startup logic
    logger.info("🚀 Starting DocSpectra RAG API...")
    
    # Validate required environment variables
    required_vars = ['PINECONE_API_KEY', 'AWS_DEFAULT_REGION', 'MARKER_S3_BUCKET']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"❌ Missing environment variables: {missing}")
        raise Exception(f"Missing environment variables: {missing}")
    
    # Here you can initialize any resources that need to be shared
    # across requests (database connections, ML models, etc.)
    
    logger.info("✅ All services ready!")
    
    # This yield separates startup from shutdown code
    yield
    
    # Shutdown logic (runs when the app is shutting down)
    logger.info("👋 Shutting down gracefully...")
    # Add any cleanup logic here (close DB connections, etc.)

# Create FastAPI app with lifespan
app = FastAPI(
    title="DocSpectra - HackRx 6.0",
    lifespan=lifespan  # Pass the lifespan context manager
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include router
app.include_router(router, prefix="/hackrx")
