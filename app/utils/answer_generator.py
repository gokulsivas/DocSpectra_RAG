# app/utils/answer_generator.py - Titan Q&A Integration with RAG
import logging
import json
from typing import List, Dict, Any
from .aws_client import aws_client
from ..config import aws_config, processing_config

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Titan Q&A integration for document processing with RAG support"""
    
    def __init__(self):
        self.model_id = aws_config.titan_model_id
        
        logger.info(f"AnswerGenerator initialized with model: {self.model_id}")
    
    def generate_answers_titan(self, ocr_text: str, questions: List[str]) -> Dict[str, Any]:
        """
        Generate answers for multiple questions using Titan Q&A with JSON output format.
        
        Args:
            ocr_text: The OCR text from the document
            questions: List of questions to answer
        
        Returns:
            Dictionary with 'answers' list or error information
        """
        try:
            # Validate inputs
            if not ocr_text or not ocr_text.strip():
                return {
                    "error": "OCR text cannot be empty",
                    "answers": []
                }
            
            if not questions or len(questions) == 0:
                return {
                    "error": "Questions list cannot be empty",
                    "answers": []
                }
            
            # Format questions block
            question_block = "\n".join([f"- {q}" for q in questions])
            
            # Build Titan prompt for JSON output
            prompt = f"""
You are an expert assistant that answers questions based only on the provided document text.

Document Content:
\"\"\"
{ocr_text.strip()}
\"\"\"

Below are several questions based on the document. Provide the answers in the following JSON format:
{{
  "answers": [
    "...", "...", ...
  ]
}}

Do not return any explanation or text outside the JSON. Just return the JSON with answers in the same order.

Questions:
{question_block}
            """.strip()
            
            # Generate answers using Bedrock Titan
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
                return {
                    "error": "Invalid response from language model",
                    "answers": []
                }
            
            result_text = result["results"][0]["outputText"]
            
            # Attempt to parse strictly as JSON
            try:
                parsed = json.loads(result_text)
                if "answers" in parsed:
                    logger.info(f"Successfully generated {len(parsed['answers'])} answers")
                    return parsed
                else:
                    logger.warning("Valid JSON but missing 'answers' key")
                    return {
                        "error": "Valid JSON but missing 'answers' key",
                        "raw_output": result_text
                    }
            except json.JSONDecodeError:
                logger.warning("Model output is not valid JSON")
                return {
                    "error": "Model output is not valid JSON",
                    "raw_output": result_text
                }
                
        except Exception as e:
            logger.error(f"Error in Titan Q&A: {str(e)}")
            return {
                "error": f"Titan Q&A failed: {str(e)}",
                "answers": []
            }
    
    def generate_answer_with_context(self, question: str, context: str) -> Dict[str, Any]:
        """
        Generate a single answer using Titan with relevant context chunks (RAG approach).
        
        Args:
            question: The question to answer
            context: Relevant context chunks from vector search
        
        Returns:
            Dictionary with 'answer' or error information
        """
        try:
            # Validate inputs
            if not question or not question.strip():
                return {
                    "error": "Question cannot be empty",
                    "answer": ""
                }
            
            if not context or not context.strip():
                return {
                    "error": "Context cannot be empty",
                    "answer": ""
                }
            
            # Build RAG prompt with context
            prompt = f"""
You are an expert assistant that answers questions based only on the provided document context.

Context from Document:
\"\"\"
{context.strip()}
\"\"\"

Question: {question.strip()}

Instructions:
- Answer the question based ONLY on the provided context
- If the answer is not in the context, clearly state that the information is not available
- Be accurate and avoid speculation
- Provide a clear, concise answer

Answer:"""
            
            # Generate answer using Bedrock Titan
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
                return {
                    "error": "Invalid response from language model",
                    "answer": ""
                }
            
            answer = result["results"][0]["outputText"].strip()
            
            # Clean up the answer
            answer = self._clean_answer(answer)
            
            logger.info(f"Generated RAG answer: {len(answer)} characters")
            return {
                "answer": answer
            }
                
        except Exception as e:
            logger.error(f"Error in RAG answer generation: {str(e)}")
            return {
                "error": f"RAG answer generation failed: {str(e)}",
                "answer": ""
            }
    
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
