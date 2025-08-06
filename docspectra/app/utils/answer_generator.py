# app/utils/answer_generator.py
import logging
from typing import List, Dict, Any
from .aws_client import aws_client
from ..config import aws_config, processing_config

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Simplified answer generation service"""
    
    def __init__(self):
        self.model_id = aws_config.titan_model_id
        self.config = processing_config
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate answer from query and context"""
        try:
            context = "\n\n".join(context_chunks)
            
            prompt = self._build_prompt(query, context)
            
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "temperature": self.config.temperature,
                    "maxTokenCount": self.config.max_tokens,
                    "topP": self.config.top_p,
                    "stopSequences": []
                }
            }
            
            result = aws_client.invoke_bedrock_model(self.model_id, body)
            answer = result["results"][0]["outputText"].strip()
            
            logger.info("Successfully generated answer")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I couldn't generate an answer due to a technical issue."
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for answer generation"""
        return f"""You are an AI assistant helping users understand documents.

Context:
{context}

Question: {query}

Answer:"""
