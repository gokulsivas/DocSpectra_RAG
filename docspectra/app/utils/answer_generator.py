# app/utils/answer_generator.py - FIXED Titan Q&A Integration
import logging
import json
from typing import List, Dict, Any
from .aws_client import aws_client
from ..config import aws_config

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Titan Q&A integration for document processing with RAG support - FIXED API"""
    
    def __init__(self):
        self.model_id = aws_config.titan_model_id
        logger.info(f"AnswerGenerator initialized with model: {self.model_id}")
    
    def generate_answer_with_context(self, question: str, context: str) -> Dict[str, Any]:
        """Generate answer with CORRECT Titan API format"""
        try:
            if not question.strip() or not context.strip():
                return {"error": "Question or context cannot be empty", "answer": ""}
            
            # Build enhanced prompt for insurance documents
            prompt = f"""You are an expert assistant that answers questions based only on the provided document context.

Context from Document:
{context.strip()}

Question: {question.strip()}

Instructions:
- Answer the question based ONLY on the provided context
- If the answer is not in the context, clearly state that the information is not available
- Be accurate and avoid speculation
- Provide a clear, concise answer
- Include specific details like numbers, percentages, time periods when available

Answer:"""
            
            # CORRECT Titan API format
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": aws_config.max_tokens,
                    "temperature": aws_config.temperature,
                    "topP": aws_config.top_p,
                    "stopSequences": []
                }
            }
            
            result = aws_client.invoke_bedrock_model(self.model_id, body)
            
            if "results" not in result or len(result["results"]) == 0:
                return {"error": "Invalid response from language model", "answer": ""}
            
            answer = result["results"][0]["outputText"].strip()
            answer = self._clean_answer(answer)
            
            logger.info(f"Generated RAG answer: {len(answer)} characters")
            return {"answer": answer}
                
        except Exception as e:
            logger.error(f"Error in RAG answer generation: {e}")
            return {"error": f"RAG answer generation failed: {str(e)}", "answer": ""}
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and validate the generated answer"""
        # Remove common artifacts
        answer = answer.replace("Answer:", "").strip()
        answer = answer.replace("Based on the context", "").strip()
        answer = answer.replace("According to the document", "").strip()
        
        # Ensure proper capitalization
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        # Limit length
        if len(answer) > 800:
            answer = answer[:797] + "..."
        
        # Validate that answer is not too short
        if len(answer.strip()) < 10:
            return "I couldn't find a specific answer to this question in the provided context."
        
        return answer