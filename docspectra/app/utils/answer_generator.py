# app/utils/answer_generator.py - Enhanced with Bedrock Titan integration
import logging
from typing import List, Dict, Any
from .aws_client import aws_client
from ..config import aws_config, processing_config

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Enhanced answer generation service with Bedrock Titan integration"""
    
    def __init__(self):
        self.model_id = aws_config.titan_model_id
        self.config = processing_config
        
        # Answer generation parameters
        self.max_context_length = 4000  # Maximum context to send to Titan
        self.max_answer_length = 800   # Maximum answer length
        
        logger.info(f"AnswerGenerator initialized with model: {self.model_id}")
        logger.info(f"Max context length: {self.max_context_length}")
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate answer from query and context using Bedrock Titan.
        
        Args:
            query: The question to answer
            context_chunks: List of relevant text chunks from the document
        
        Returns:
            Generated answer string
        """
        try:
            # Validate inputs
            if not query or not query.strip():
                return "Error: Question cannot be empty."
            
            if not context_chunks:
                return "I don't have enough context information to answer this question."
            
            # Prepare and optimize context
            context = self._prepare_context(context_chunks)
            
            if not context.strip():
                return "I couldn't find relevant information to answer this question."
            
            # Build enhanced prompt
            prompt = self._build_enhanced_prompt(query.strip(), context)
            
            # Generate answer using Bedrock Titan
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "temperature": self.config.temperature,
                    "maxTokenCount": self.config.max_tokens,
                    "topP": self.config.top_p,
                    "stopSequences": [
                        "Human:", "Context:", "Question:", 
                        "\n\n---", "END_ANSWER", "[Context"
                    ]
                }
            }
            
            result = aws_client.invoke_bedrock_model(self.model_id, body)
            
            if "results" not in result or len(result["results"]) == 0:
                return "Error: Invalid response from language model."
            
            answer = result["results"][0]["outputText"].strip()
            
            # Post-process and clean the answer
            answer = self._clean_and_validate_answer(answer, query)
            
            logger.info(f"Generated answer: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}"
    
    def _prepare_context(self, context_chunks: List[str]) -> str:
        """
        Prepare and optimize context from chunks for better answer generation.
        
        Args:
            context_chunks: List of text chunks
        
        Returns:
            Optimized context string
        """
        # Remove empty chunks and duplicates
        valid_chunks = []
        seen_chunks = set()
        
        for chunk in context_chunks:
            chunk = chunk.strip()
            if chunk and chunk not in seen_chunks:
                valid_chunks.append(chunk)
                seen_chunks.add(chunk)
        
        if not valid_chunks:
            return ""
        
        # Join chunks with clear separators
        context = "\n\n---\n\n".join(valid_chunks)
        
        # Truncate if too long, keeping whole chunks when possible
        if len(context) > self.max_context_length:
            truncated_context = ""
            current_length = 0
            
            for chunk in valid_chunks:
                chunk_with_separator = chunk + "\n\n---\n\n"
                if current_length + len(chunk_with_separator) <= self.max_context_length:
                    truncated_context += chunk_with_separator
                    current_length += len(chunk_with_separator)
                else:
                    break
