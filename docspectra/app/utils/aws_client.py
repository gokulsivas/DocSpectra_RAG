# app/utils/aws_client.py - Enhanced for Bedrock Titan integration
import boto3
import json
import logging
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional, Dict, Any, List
from ..config import aws_config

logger = logging.getLogger(__name__)

class AWSClientManager:
    """Enhanced AWS client management with comprehensive Bedrock support"""
    
    def __init__(self):
        self._bedrock_client = None
        self._bedrock_agent_client = None
        self._s3_client = None
        self._credentials_validated = False
        
        # Bedrock configuration
        self.titan_text_model = aws_config.titan_model_id
        self.titan_embed_model = aws_config.titan_embed_model
        self.region = aws_config.region
        
        logger.info(f"AWSClientManager initialized for region: {self.region}")
        logger.info(f"Titan text model: {self.titan_text_model}")
        logger.info(f"Titan embed model: {self.titan_embed_model}")
    
    def _validate_credentials(self):
        """Validate AWS credentials and permissions"""
        if self._credentials_validated:
            return True
            
        try:
            # Test with STS get-caller-identity
            sts = boto3.client('sts', region_name=self.region)
            identity = sts.get_caller_identity()
            
            account_id = identity.get('Account', 'unknown')
            user_arn = identity.get('Arn', 'unknown')
            
            logger.info(f"AWS credentials validated successfully")
            logger.info(f"Account: {account_id}")
            logger.info(f"User/Role: {user_arn}")
            
            self._credentials_validated = True
            return True
            
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"AWS credential validation failed: {e}")
            
            # Provide helpful error messages
            error_msg = "AWS credentials not configured properly. Please ensure:\n"
            error_msg += "1. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
            error_msg += "2. Or configure AWS CLI: aws configure\n"
            error_msg += "3. Or use IAM roles if running on EC2\n"
            error_msg += "4. Ensure credentials have Bedrock permissions"
            
            raise Exception(error_msg)
    
    @property
    def bedrock_client(self):
        """Lazy-loaded Bedrock runtime client"""
        if self._bedrock_client is None:
            self._validate_credentials()
            self._bedrock_client = boto3.client(
                'bedrock-runtime', 
                region_name=self.region
            )
            logger.info("Bedrock runtime client initialized successfully")
        return self._bedrock_client
    
    @property
    def bedrock_agent_client(self):
        """Lazy-loaded Bedrock agent client (for model management)"""
        if self._bedrock_agent_client is None:
            self._validate_credentials()
            self._bedrock_agent_client = boto3.client(
                'bedrock',
                region_name=self.region
            )
            logger.info("Bedrock agent client initialized successfully")
        return self._bedrock_agent_client
    
    @property
    def s3_client(self):
        """Lazy-loaded S3 client"""
        if self._s3_client is None:
            self._validate_credentials()
            self._s3_client = boto3.client(
                's3',
                region_name=self.region
            )
        return self._s3_client
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available Bedrock foundation models"""
        try:
            response = self.bedrock_agent_client.list_foundation_models()
            models = response.get('modelSummaries', [])
            
            # Filter for Titan models
            titan_models = [
                model for model in models 
                if 'titan' in model.get('modelId', '').lower()
            ]
            
            logger.info(f"Found {len(titan_models)} Titan models available:")
            for model in titan_models:
                model_id = model.get('modelId', 'unknown')
                status = model.get('modelLifecycle', {}).get('status', 'unknown')
                logger.info(f"  - {model_id} (Status: {status})")
            
            return titan_models
            
        except Exception as e:
            logger.error(f"Error listing Bedrock models: {e}")
            return []
    
    def invoke_bedrock_model(self, model_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified Bedrock model invocation with comprehensive error handling.
        
        Args:
            model_id: Bedrock model ID (e.g., 'amazon.titan-text-express-v1')
            body: Request body as dictionary
        
        Returns:
            Response from Bedrock model
        """
        try:
            logger.debug(f"Invoking Bedrock model: {model_id}")
            logger.debug(f"Request body keys: {list(body.keys())}")
            
            # Validate model ID
            if not model_id or not isinstance(model_id, str):
                raise ValueError(f"Invalid model ID: {model_id}")
            
            # Prepare request
            request_body = json.dumps(body).encode("utf-8")
            
            # Invoke model
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=request_body
            )
            
            # Parse response
            result = json.loads(response['body'].read().decode())
            
            logger.debug(f"Model invocation successful for {model_id}")
            logger.debug(f"Response keys: {list(result.keys())}")
            
            return result
        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            # Handle specific error cases with helpful messages
            if error_code == 'ValidationException':
                logger.error(f"Validation error for {model_id}: {error_message}")
                raise ValueError(f"Invalid request for model {model_id}: {error_message}")
                
            elif error_code == 'ResourceNotFoundException':
                logger.error(f"Model not found: {model_id}")
                available_models = [m['modelId'] for m in self.list_available_models()]
                raise ValueError(
                    f"Model {model_id} not available in region {self.region}. "
                    f"Available Titan models: {available_models}"
                )
                
            elif error_code == 'AccessDeniedException':
                logger.error(f"Access denied for model {model_id}")
                raise PermissionError(
                    f"Access denied for model {model_id}. Please ensure:\n"
                    f"1. Model access is enabled in AWS Bedrock console\n"
                    f"2. IAM policy includes 'bedrock:InvokeModel' permission\n"
                    f"3. Resource ARN includes the model: arn:aws:bedrock:{self.region}::foundation-model/{model_id}"
                )
                
            elif error_code == 'ThrottlingException':
                logger.error(f"Rate limit exceeded for {model_id}")
                raise Exception(f"Rate limit exceeded for model {model_id}. Please implement retry logic with exponential backoff.")
                
            elif error_code == 'ServiceQuotaExceededException':
                logger.error(f"Service quota exceeded for {model_id}")
                raise Exception(f"Service quota exceeded for model {model_id}. Please check your AWS service quotas.")
                
            else:
                logger.error(f"Bedrock error for {model_id}: {error_code} - {error_message}")
                raise Exception(f"Bedrock API error [{error_code}]: {error_message}")
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for response from {model_id}: {e}")
            raise Exception(f"Invalid JSON response from model {model_id}")
        
        except Exception as e:
            logger.error(f"Unexpected error invoking {model_id}: {e}")
            raise
    
    def invoke_titan_text_model(self, prompt: str, **kwargs) -> str:
        """
        Convenience method for Titan text model invocation.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text response
        """
        # Default parameters
        default_params = {
            "temperature": 0.7,
            "maxTokenCount": 300,
            "topP": 0.9,
            "stopSequences": []
        }
        
        # Update with provided parameters
        generation_config = {**default_params, **kwargs}
        
        # Prepare request body
        body = {
            "inputText": prompt,
            "textGenerationConfig": generation_config
        }
        
        # Invoke model
        result = self.invoke_bedrock_model(self.titan_text_model, body)
        
        # Extract text from response
        if "results" in result and len(result["results"]) > 0:
            return result["results"][0]["outputText"]
        else:
            raise Exception("Invalid response format from Titan text model")
    
    def invoke_titan_embed_model(self, text: str) -> List[float]:
        """
        Convenience method for Titan embedding model invocation.
        
        Args:
            text: Input text to embed
        
        Returns:
            Embedding vector as list of floats
        """
        # Prepare request body
        body = {"inputText": text}
        
        # Invoke model
        result = self.invoke_bedrock_model(self.titan_embed_model, body)
        
        # Extract embedding from response
        if "embedding" in result:
            return result["embedding"]
        else:
            raise Exception("Invalid response format from Titan embedding model")
    
    def test_titan_text_model(self) -> bool:
        """Test Titan text generation model"""
        try:
            logger.info("Testing Titan text model...")
            
            test_prompt = "What is artificial intelligence? Provide a brief explanation."
            response = self.invoke_titan_text_model(
                prompt=test_prompt,
                maxTokenCount=100,
                temperature=0.7
            )
            
            if response and len(response.strip()) > 10:
                logger.info(f"✅ Titan text model test successful")
                logger.info(f"Response preview: {response[:100]}...")
                return True
            else:
                logger.error("❌ Titan text model returned empty or invalid response")
                return False
                
        except Exception as e:
            logger.error(f"❌ Titan text model test failed: {e}")
            return False
    
    def test_titan_embed_model(self) -> bool:
        """Test Titan embedding model"""
        try:
            logger.info("Testing Titan embedding model...")
            
            test_text = "This is a test sentence for embedding generation."
            embedding = self.invoke_titan_embed_model(test_text)
            
            if embedding and len(embedding) > 0:
                embedding_length = len(embedding)
                logger.info(f"✅ Titan embedding model test successful")
                logger.info(f"Generated embedding with {embedding_length} dimensions")
                return True
            else:
                logger.error("❌ Titan embedding model returned empty embedding")
                return False
                
        except Exception as e:
            logger.error(f"❌ Titan embedding model test failed: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive Bedrock health check"""
        health_status = {
            "overall_status": False,
            "aws_credentials": False,
            "bedrock_access": False,
            "titan_text_model": False,
            "titan_embed_model": False,
            "available_models": [],
            "region": self.region,
            "errors": []
        }
        
        try:
            # Test 1: AWS credentials
            logger.info("🔍 Checking AWS credentials...")
            self._validate_credentials()
            health_status["aws_credentials"] = True
            logger.info("✅ AWS credentials valid")
            
            # Test 2: Bedrock access and model listing
            logger.info("🔍 Checking Bedrock access...")
            models = self.list_available_models()
            health_status["available_models"] = [m["modelId"] for m in models]
            health_status["bedrock_access"] = len(models) > 0
            
            if health_status["bedrock_access"]:
                logger.info(f"✅ Bedrock access confirmed ({len(models)} Titan models)")
            else:
                logger.warning("⚠️ No Titan models found - check model access in Bedrock console")
            
            # Test 3: Titan text model
            if self.titan_text_model in health_status["available_models"]:
                logger.info("🔍 Testing Titan text model...")
                health_status["titan_text_model"] = self.test_titan_text_model()
            else:
                health_status["errors"].append(f"Titan text model {self.titan_text_model} not available")
                logger.error(f"❌ Titan text model {self.titan_text_model} not in available models")
            
            # Test 4: Titan embedding model
            if self.titan_embed_model in health_status["available_models"]:
                logger.info("🔍 Testing Titan embedding model...")
                health_status["titan_embed_model"] = self.test_titan_embed_model()
            else:
                health_status["errors"].append(f"Titan embed model {self.titan_embed_model} not available")
                logger.error(f"❌ Titan embedding model {self.titan_embed_model} not in available models")
                
        except Exception as e:
            health_status["errors"].append(str(e))
            logger.error(f"❌ Health check error: {e}")
        
        # Calculate overall status
        health_status["overall_status"] = (
            health_status["aws_credentials"] and 
            health_status["bedrock_access"] and
            health_status["titan_text_model"] and 
            health_status["titan_embed_model"]
        )
        
        # Log final status
        if health_status["overall_status"]:
            logger.info("🎉 All Bedrock health checks passed!")
        else:
            logger.warning("⚠️ Some Bedrock health checks failed")
            for error in health_status["errors"]:
                logger.warning(f"   - {error}")
        
        return health_status
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        try:
            response = self.bedrock_agent_client.get_foundation_model(modelIdentifier=model_id)
            return response.get('modelDetails', {})
        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {e}")
            return {}

# Global client manager instance
aws_client = AWSClientManager()

# Convenience functions for easy access
def invoke_titan_text(prompt: str, **kwargs) -> str:
    """Global convenience function for Titan text generation"""
    return aws_client.invoke_titan_text_model(prompt, **kwargs)

def invoke_titan_embed(text: str) -> List[float]:
    """Global convenience function for Titan embedding"""
    return aws_client.invoke_titan_embed_model(text)

def bedrock_health_check() -> Dict[str, Any]:
    """Global convenience function for health check"""
    return aws_client.health_check()
