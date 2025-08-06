#!/usr/bin/env python3
"""
Startup script for DocSpectra RAG API with environment validation
"""

import os
import sys
from dotenv import load_dotenv
import uvicorn

def validate_environment():
    """Validate that all required environment variables are set"""
    print("🔍 Validating environment variables...")
    
    # Load .env file
    load_dotenv()
    
    # Required environment variables
    required_vars = {
        'PINECONE_API_KEY': 'Pinecone API Key',
        'MARKER_S3_BUCKET': 'Marker S3 Bucket',
        'AWS_DEFAULT_REGION': 'AWS Region'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease check your .env file and ensure all required variables are set.")
        return False
    
    print("✅ All required environment variables are set!")
    return True

def start_server():
    """Start the FastAPI server"""
    if not validate_environment():
        sys.exit(1)
    
    # Get configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print(f"\n🚀 Starting DocSpectra RAG API...")
    print(f"📍 Server: http://{host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔧 Health Check: http://{host}:{port}/hackrx/health")
    print(f"🔄 Auto-reload: {'Enabled' if reload else 'Disabled'}")
    print("\n" + "="*50)
    
    try:
        # Start the server
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server() 