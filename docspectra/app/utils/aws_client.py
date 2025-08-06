# app/utils/aws_client.py
import boto3
import json
import logging
from botocore.exceptions import ClientError
from typing import Optional, Dict, Any
from ..config import aws_config

logger = logging.getLogger(__name__)

class AWSClientManager:
    """Centralized AWS client management"""
    
    def __init__(self):
        self._bedrock_client = None
        self._s3_client = None
    
    @property
    def bedrock_client(self):
        """Lazy-loaded Bedrock client"""
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client(
                'bedrock-runtime', 
                region_name=aws_config.region
            )
        return self._bedrock_client
    
    @property
    def s3_client(self):
        """Lazy-loaded S3 client"""
        if self._s3_client is None:
            self._s3_client = boto3.client(
                's3',
                region_name=aws_config.region
            )
        return self._s3_client
    
    def invoke_bedrock_model(self, model_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Unified Bedrock model invocation with error handling"""
        try:
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body).encode("utf-8")
            )
            return json.loads(response['body'].read().decode())
        
        except ClientError as e:
            logger.error(f"Bedrock invoke_model failed for {model_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error invoking {model_id}: {e}")
            raise

# Global client manager instance
aws_client = AWSClientManager()
