# app/utils/answer_generator.py - Enhanced with better prompting
import logging
import json
from typing import List, Dict, Any
from .aws_client import aws_client
from ..config import aws_config

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Enhanced Titan Q&A integration with better prompting"""
    
    def __init__(self):
        self.model_id = aws_config.titan_model_id
        logger.info(f"AnswerGenerator initialized with model: {self.model_id}")
    
    def generate_answer_with_context(self, question: str, context: str) -> Dict[str, Any]:
        """Generate answer with enhanced prompting for insurance documents"""
        try:
            if not question.strip() or not context.strip():
                return {"error": "Question or context cannot be empty", "answer": ""}
            
            # Enhanced prompt for insurance documents
            prompt = f"""You are an expert insurance policy analyst. Answer the question based ONLY on the provided policy document context.

CONTEXT FROM INSURANCE POLICY:
{context.strip()}

QUESTION: {question.strip()}

INSTRUCTIONS:
1. Answer based ONLY on the information provided in the context above
2. Be specific and include relevant details like numbers, percentages, time periods
3. If the exact information isn't in the context, say "The specific information is not provided in this section of the policy"
4. For insurance terms, provide clear explanations
5. Include relevant conditions or limitations mentioned in the context
6. Do not make assumptions or add information not in the context

ANSWER:"""
            
            # Generate answer using Bedrock Titan
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": min(aws_config.max_tokens, 1000),  # Increase token limit
                    "temperature": 0.1,  # Lower temperature for more precise answers
                    "topP": aws_config.top_p,
                    "stopSequences": ["CONTEXT:", "QUESTION:", "INSTRUCTIONS:"]
                }
            }
            
            result = aws_client.invoke_bedrock_model(self.model_id, body)
            
            if "results" not in result or len(result["results"]) == 0:
                return {"error": "Invalid response from language model", "answer": ""}
            
            answer = result["results"][0]["outputText"].strip()
            answer = self._clean_answer(answer)
            
            logger.info(f"Generated enhanced answer: {len(answer)} characters")
            return {"answer": answer}
                
        except Exception as e:
            logger.error(f"Error in enhanced answer generation: {e}")
            return {"error": f"Answer generation failed: {str(e)}", "answer": ""}
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and validate the generated answer"""
        # Remove common artifacts
        answer = answer.replace("ANSWER:", "").strip()
        answer = answer.replace("Based on the context", "").strip()
        answer = answer.replace("According to the policy", "").strip()
        
        # Ensure proper capitalization
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        # Don't truncate insurance answers - they often need full detail
        if len(answer.strip()) < 10:
            return "The specific information requested is not available in the provided policy section."
        
        return answer