# DocSpectra RAG System

A clean, optimized LLM-powered intelligent query-retrieval system for processing large documents and making contextual decisions.

## 🚀 Quick Start

### Prerequisites

1. **Environment Variables** - Create a `.env` file in the `docspectra` directory:

```env
# AWS Configuration
AWS_DEFAULT_REGION=us-east-1
TITAN_MODEL_ID=amazon.titan-text-express-v1
TITAN_EMBED_MODEL=amazon.titan-embed-text-v1
MARKER_S3_BUCKET=your-marker-models-bucket

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENV=us-east-1

# Optional Configuration
CHUNK_MAX_WORDS=150
MAX_TOKENS=300
TEMPERATURE=0.7
TOP_P=0.9
TOP_K_RETRIEVAL=5
```

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
# Option 1: Using the run script
python run.py

# Option 2: Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 📚 API Usage

### Health Check
```bash
curl http://localhost:8000/hackrx/health
```

### Process Document and Answer Questions
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What is the waiting period for pre-existing diseases?"
    ]
  }'
```

## 🏗️ System Architecture

The cleaned up system includes:

1. **Document Processing**: Marker PDF parser with S3-cached models
2. **Text Chunking**: Sentence-aware text chunking with overlap
3. **Embedding Generation**: AWS Titan embedding model
4. **Vector Storage**: Pinecone vector database
5. **Answer Generation**: AWS Titan text model for intelligent Q&A

## 🧹 What Was Cleaned Up

- ❌ Removed unnecessary model listing functions
- ❌ Removed redundant S3 model manager files
- ❌ Removed unused deployment scripts
- ❌ Simplified health checks
- ✅ Optimized for Titan Lite Express only
- ✅ Lazy initialization to avoid startup errors
- ✅ Clean, focused codebase

## 🔧 Key Features

- **Efficient Processing**: Uses S3-cached Marker models
- **Intelligent Q&A**: Bedrock Titan integration for accurate answers
- **Vector Search**: Pinecone for semantic similarity
- **Error Handling**: Comprehensive error handling and logging
- **Scalable**: Designed for production deployment

## 📖 API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation. 