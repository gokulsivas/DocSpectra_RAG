#!/usr/bin/env python3
"""
Simple test script to verify API structure
"""

import requests
import json
from app.main import app
from fastapi.testclient import TestClient

def test_api_structure():
    """Test the API structure and endpoints"""
    client = TestClient(app)
    
    print("🧪 Testing API structure...")
    
    # Test health endpoint
    try:
        response = client.get("/hackrx/health")
        print(f"✅ Health endpoint: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
    
    # Test API documentation
    try:
        response = client.get("/docs")
        print(f"✅ API docs endpoint: {response.status_code}")
    except Exception as e:
        print(f"❌ API docs failed: {e}")
    
    # Test OpenAPI schema
    try:
        response = client.get("/openapi.json")
        print(f"✅ OpenAPI schema: {response.status_code}")
    except Exception as e:
        print(f"❌ OpenAPI schema failed: {e}")
    
    print("✅ API structure test completed!")

if __name__ == "__main__":
    test_api_structure() 