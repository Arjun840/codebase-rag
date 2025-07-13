"""Answer generator for the RAG system."""

import logging
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)


class Generator:
    """Generates answers using a language model."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", max_length: int = 512):
        """Initialize the generator."""
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
    
    async def initialize(self):
        """Initialize the generation model."""
        if self.model is not None:
            return
        
        logger.info(f"Loading generation model: {self.model_name}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move to device
            self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                do_sample=True,
                temperature=0.7,
                max_length=self.max_length,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Generation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load generation model: {e}")
            raise
    
    async def generate_answer(self, query: str, context: str, max_new_tokens: int = 200) -> str:
        """Generate an answer given a query and context."""
        if self.pipeline is None:
            await self.initialize()
        
        # Prepare the prompt
        prompt = self._prepare_prompt(query, context)
        
        try:
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Clean up the response
            answer = self._clean_response(generated_text)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I apologize, but I encountered an error while generating an answer: {str(e)}"
    
    async def generate_code_explanation(self, code_snippet: str, context: str = "") -> str:
        """Generate an explanation for a code snippet."""
        query = f"Explain this code snippet: {code_snippet}"
        return await self.generate_answer(query, context)
    
    async def generate_error_solution(self, error_message: str, context: str = "") -> str:
        """Generate a solution for an error message."""
        query = f"How to fix this error: {error_message}"
        return await self.generate_answer(query, context)
    
    async def generate_code_suggestion(self, description: str, context: str = "") -> str:
        """Generate code suggestions based on description."""
        query = f"Write code to: {description}"
        return await self.generate_answer(query, context)
    
    def _prepare_prompt(self, query: str, context: str) -> str:
        """Prepare the prompt for the language model."""
        # Create a structured prompt
        prompt_parts = [
            "You are a helpful coding assistant that answers questions about code and programming.",
            "Use the following context to answer the question.",
            "",
            "Context:",
            context,
            "",
            "Question:",
            query,
            "",
            "Answer:"
        ]
        
        prompt = "\n".join(prompt_parts)
        
        # Truncate if too long
        if len(prompt) > self.max_length * 3:  # Rough estimate
            # Truncate context while keeping the structure
            max_context_length = self.max_length * 2
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
                prompt_parts[4] = context
                prompt = "\n".join(prompt_parts)
        
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response."""
        # Remove extra whitespace
        response = response.strip()
        
        # Remove common artifacts
        response = response.replace("<|endoftext|>", "")
        response = response.replace("</s>", "")
        
        # Stop at certain tokens that might indicate end of response
        stop_tokens = ["Human:", "Question:", "Context:", "Query:"]
        for token in stop_tokens:
            if token in response:
                response = response.split(token)[0].strip()
        
        # If response is too short, add a helpful message
        if len(response.strip()) < 10:
            response = "I'm sorry, but I couldn't generate a comprehensive answer based on the provided context. Could you please provide more specific information or rephrase your question?"
        
        return response
    
    async def generate_with_temperature(self, query: str, context: str, temperature: float = 0.7) -> str:
        """Generate answer with custom temperature."""
        if self.pipeline is None:
            await self.initialize()
        
        # Prepare prompt
        prompt = self._prepare_prompt(query, context)
        
        try:
            # Generate with custom temperature
            response = self.pipeline(
                prompt,
                max_new_tokens=200,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            generated_text = response[0]['generated_text']
            return self._clean_response(generated_text)
            
        except Exception as e:
            logger.error(f"Error generating answer with temperature {temperature}: {e}")
            return f"I apologize, but I encountered an error while generating an answer: {str(e)}"
    
    async def generate_multiple_answers(self, query: str, context: str, num_answers: int = 3) -> List[str]:
        """Generate multiple answer variations."""
        answers = []
        
        for i in range(num_answers):
            # Use different temperatures for variety
            temperature = 0.5 + (i * 0.3)
            answer = await self.generate_with_temperature(query, context, temperature)
            answers.append(answer)
        
        return answers
    
    async def cleanup(self):
        """Cleanup generator resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Generator cleanup completed") 