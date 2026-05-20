#!/usr/bin/env python3
"""
Test script for DocSpectra RAG API with Vector DB
"""

import requests
import json
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_document_processing():
    """Test the document processing endpoint with RAG"""
    
    # Test data
    test_url = "https://example.com/sample-policy.pdf"  # Replace with actual PDF URL
    test_questions = [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?",
        "What is the waiting period for cataract surgery?",
        "Are medical expenses for organ donors covered?"
    ]
    
    # Prepare request
    payload = {
        "documents": test_url,
        "questions": test_questions
    }
    
    print("🧪 Testing RAG Document Processing Pipeline...")
    print(f"📄 Document URL: {test_url}")
    print(f"❓ Questions: {len(test_questions)}")
    print("\n" + "="*50)
    
    try:
        # Make request to the RAG endpoint
        response = requests.post(
            "http://localhost:8000/hackrx/run",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            answers = result['answers']  # Get answers from wrapped object
            print("✅ RAG document processing test successful!")
            print(f"📝 Generated {len(answers)} answers:")
            
            for i, answer in enumerate(answers, 1):
                print(f"\n{i}. {answer[:100]}...")
                
        else:
            print(f"❌ RAG document processing test failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

def test_single_question():
    """Test with a single question to verify RAG works for individual queries"""
    
    # Test data
    test_url = "https://example.com/sample-policy.pdf"  # Replace with actual PDF URL
    test_questions = [
        "What is the grace period for premium payment?"
    ]
    
    # Prepare request
    payload = {
        "documents": test_url,
        "questions": test_questions
    }
    
    print("\n🧪 Testing Single Question RAG...")
    print(f"📄 Document URL: {test_url}")
    print(f"❓ Question: {test_questions[0]}")
    print("\n" + "="*50)
    
    try:
        # Make request to the RAG endpoint
        response = requests.post(
            "http://localhost:8000/hackrx/run",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            answers = result['answers']  # Get answers from wrapped object
            print("✅ Single question RAG test successful!")
            print(f"📝 Answer: {answers[0]}")
                
        else:
            print(f"❌ Single question RAG test failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

def test_health_check():
    """Test the health check endpoint"""
    
    print("\n🧪 Testing Health Check...")
    print("="*50)
    
    try:
        response = requests.get("http://localhost:8000/hackrx/health")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check successful!")
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Service: {result.get('service', 'unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 DocSpectra RAG API Test (with Vector DB)")
    print("="*50)
    
    # Test health check first
    test_health_check()
    
    # Test single question
    test_single_question()
    
    # Test multiple questions
    test_document_processing()
    
    print("\n" + "="*50)
    print("✅ Testing completed!")
    print("\n📋 Usage:")
    print("1. Start server: python -m uvicorn app.main:app --reload")
    print("2. Test health: GET /hackrx/health")
    print("3. Process document: POST /hackrx/run")
    print("\n📋 RAG Pipeline:")
    print("Document → Marker → Chunks → VectorDB → Search → Titan → JSON Answers")
    print("\n📋 Output Format:")
    print("Returns: {\"answers\": [\"answer1\", \"answer2\", ...]}")
