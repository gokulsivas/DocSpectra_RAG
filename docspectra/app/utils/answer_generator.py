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
            
            context = truncated_context
        
        return context
    
    def _build_enhanced_prompt(self, query: str, context: str) -> str:
        """
        Build enhanced prompt for better answer generation.
        
        Args:
            query: The question to answer
            context: Relevant context from document
        
        Returns:
            Formatted prompt string
        """
        return f"""You are an AI assistant that answers questions based on provided document context. 

Instructions:
- Answer the question directly and concisely based ONLY on the provided context
- If the answer is not in the context, clearly state that the information is not available
- Be accurate and avoid speculation or information not in the context
- Keep your answer focused and relevant to the question
- Use clear, professional language
- If the question asks about specific conditions, requirements, or limitations, be sure to mention them

Context:
{context}

Question: {query}

Answer:"""
    
    def _clean_and_validate_answer(self, answer: str, query: str) -> str:
        """
        Clean and validate the generated answer.
        
        Args:
            answer: Raw answer from model
            query: Original query for validation
        
        Returns:
            Cleaned answer string
        """
        # Remove common artifacts
        answer = answer.replace("Answer:", "").strip()
        answer = answer.replace("Based on the context", "").strip()
        answer = answer.replace("According to the document", "").strip()
        
        # Remove incomplete sentences at the end
        sentences = answer.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            answer = '.'.join(sentences[:-1]) + '.'
        
        # Ensure proper capitalization
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        # Limit length
        if len(answer) > self.max_answer_length:
            answer = answer[:self.max_answer_length-3] + "..."
        
        # Validate that answer is not too short
        if len(answer.strip()) < 10:
            return "I couldn't find a specific answer to this question in the provided context."
        
        return answer
