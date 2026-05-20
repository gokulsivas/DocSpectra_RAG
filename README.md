<div align="center">

# DocSpectra RAG

**Cloud-native document Q&A powered by AWS Bedrock - built for HackRx 6.0.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![AWS Bedrock](https://img.shields.io/badge/AWS_Bedrock-Titan-FF9900?style=flat&logo=amazonaws&logoColor=white)](https://aws.amazon.com/bedrock/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-00B8D9?style=flat)](https://www.pinecone.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

DocSpectra is a cloud-native Retrieval-Augmented Generation (RAG) pipeline that accepts a document URL, extracts and semantically chunks its content, stores embeddings in Pinecone, and answers natural-language questions using Amazon Titan via AWS Bedrock, entirely on AWS infrastructure.

---

## Features

- **Document Ingestion via URL** - Accepts any publicly accessible PDF URL, downloads and stores it in S3
- **Marker-based OCR** - Extracts clean structured text from PDFs using S3-cached Marker models
- **Sentence-aware Chunking** - Splits text at sentence boundaries with 20-word overlap to preserve context
- **Titan Embeddings** - Generates 1536-dimensional vectors using Amazon Titan Embed Text v1 via Bedrock
- **Pinecone Vector Store** - Serverless cosine-similarity index with batched upsert (100 vectors/batch)
- **RAG Answer Generation** - Retrieves top-k relevant chunks per question and grounds answers using Titan LLM
- **Lazy Initialization** - Services initialize on first use to avoid startup errors
- **HackRx 6.0 API Format** - Single `/hackrx/run` endpoint; accepts `documents` URL + `questions` list, returns `answers` list

---

## Architecture

```
Document URL
     |
     v
[Marker Handler]  ---- S3 Bucket (PDF storage + OCR output)
     |
     v
[Text Chunker]    ---- Sentence-boundary chunking with 20-word overlap
     |
     v
[Titan Embedder]  ---- AWS Bedrock (amazon.titan-embed-text-v1)
     |
     v
[Pinecone Vector Store]  ---- Serverless index, cosine similarity
     |
     v
[RAG Query Engine]  ---- Top-k retrieval + context injection
     |
     v
[Titan LLM]       ---- AWS Bedrock (amazon.titan-text-express-v1)
     |
     v
Structured JSON Answers
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| LLM | AWS Bedrock - Amazon Titan Text Express v1 |
| Embeddings | AWS Bedrock - Amazon Titan Embed Text v1 |
| Vector Store | Pinecone Serverless (AWS us-east-1, cosine) |
| Document Storage | AWS S3 |
| OCR / Parsing | Marker |
| Cloud | AWS EC2 + Bedrock + S3 |
| Language | Python 3.10+ |

---

## Project Structure

```
DocSpectra_RAG/
├── app/
│   ├── main.py                   # FastAPI app with async lifespan management
│   ├── router.py                 # API endpoints (/hackrx/run, /hackrx/health)
│   ├── config.py                 # AWS, Pinecone, and processing config
│   └── utils/
│       ├── aws_client.py         # Boto3 manager for Bedrock + S3
│       ├── answer_generator.py   # Titan Q&A and RAG answer generation
│       ├── chunker.py            # Sentence-boundary chunker with overlap
│       ├── document_processor.py # End-to-end pipeline orchestrator
│       ├── embedder.py           # Titan embedding wrapper
│       ├── marker_handler.py     # PDF parsing via Marker + S3 integration
│       └── vector_store.py       # Pinecone CRUD and similarity search
├── run.sh                        # Server startup script
├── test_api.py                   # API endpoint tests
└── test_titan_qa.py              # Titan Q&A integration tests
```

---

## API Endpoints

### `POST /hackrx/run`

Processes a document URL and returns answers to a list of questions.

**Request Body:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The grace period is ...",
    "The waiting period is ..."
  ]
}
```

### `GET /hackrx/health`

Returns system health and available endpoints.

> Once the server is running, visit `http://localhost:8000/docs` for the full interactive API documentation.

---

## Setup

### Prerequisites

- Python 3.10+
- AWS account with Bedrock model access enabled for:
  - `amazon.titan-text-express-v1`
  - `amazon.titan-embed-text-v1`
- Pinecone account (Serverless tier)
- S3 bucket for document storage

### 1. Clone the Repository

```bash
git clone https://github.com/gokulsivas/DocSpectra_RAG.git
cd DocSpectra_RAG
```

### 2. Install Dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# AWS
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
TITAN_MODEL_ID=amazon.titan-text-express-v1
TITAN_EMBED_MODEL=amazon.titan-embed-text-v1

# S3
MARKER_S3_BUCKET=your-s3-bucket-name

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=us-east-1

# Optional - Processing Config
CHUNK_MAX_WORDS=150
MAX_TOKENS=300
TEMPERATURE=0.7
TOP_P=0.9
TOP_K_RETRIEVAL=5
```

### 4. Run the Server

```bash
# Option 1 - Using the run script
bash run.sh

# Option 2 - Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## AWS IAM Permissions

The IAM user or role needs the following minimum permissions:

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "s3:PutObject",
    "s3:GetObject",
    "s3:ListBucket"
  ],
  "Resource": "*"
}
```

---

## Running Tests

```bash
python test_titan_qa.py    # End-to-end Titan Q&A integration test
python test_api.py         # API endpoint tests
```

---

## Deployment Note

This project was built and deployed on **AWS EC2** with documents stored in **AWS S3** and vectors managed via **Pinecone Serverless**. The AWS infrastructure has since been decommissioned to avoid ongoing costs. All source code and architecture remain fully documented here.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  Built by <a href="https://github.com/gokulsivas">Gokul Sivasubramaniam</a>
</div>
