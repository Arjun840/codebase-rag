"""Answer generator for the RAG system."""

import logging
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import openai
import os

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
        self.is_api_model = model_name.startswith(('openai/', 'gpt-'))
        self.is_hf_api_model = model_name.startswith('huggingface/')
        self.is_ollama_model = model_name.startswith('ollama/')
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Is API model: {self.is_api_model}")
        logger.info(f"Is HF API model: {self.is_hf_api_model}")
        logger.info(f"Is Ollama model: {self.is_ollama_model}")
        logger.info(f"Model type: {'API' if self.is_api_model else 'Ollama' if self.is_ollama_model else 'Local'}")
    
    async def initialize(self):
        """Initialize the generation model."""
        if self.model is not None or self.is_api_model or self.is_ollama_model or self.is_hf_api_model:
            logger.info(f"Skipping local model loading for {self.model_name} (API or external model)")
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
        if self.is_api_model:
            return await self._generate_with_api(query, context, max_new_tokens)
        elif self.is_hf_api_model:
            return await self._generate_with_hf_api(query, context, max_new_tokens)
        elif self.is_ollama_model:
            return await self._generate_with_ollama(query, context, max_new_tokens)
        else:
            if self.pipeline is None:
                await self.initialize()
            return await self._generate_with_local(query, context, max_new_tokens)
    
    async def _generate_with_api(self, query: str, context: str, max_new_tokens: int = 200) -> str:
        """Generate answer using OpenAI API."""
        try:
            # Get API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Extract model name (remove 'openai/' prefix if present)
            model_name = self.model_name.replace('openai/', '')
            
            # Prepare the prompt
            prompt = self._prepare_prompt(query, context)
            
            # Generate response
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant that answers questions about code and programming. Use the provided context to give accurate and helpful answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_new_tokens,
                temperature=0.7
            )
            
            # Extract generated text
            generated_text = response.choices[0].message.content
            
            # Clean up the response
            answer = self._clean_response(generated_text)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with API: {e}")
            return f"I apologize, but I encountered an error while generating an answer: {str(e)}"
    
    async def _generate_with_ollama(self, query: str, context: str, max_new_tokens: int = 200) -> str:
        """Generate answer using Ollama API with enhanced post-processing."""
        try:
            import requests
            import json
            import re
            
            # Extract model name (remove 'ollama/' prefix)
            model_name = self.model_name.replace('ollama/', '')
            
            # Prepare the prompt
            prompt = self._prepare_prompt(query, context)
            
            # Ollama API endpoint
            api_url = "http://localhost:11434/api/generate"
            
            # Payload with optimized parameters for code generation
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_new_tokens,
                    "temperature": 0.3,  # Lower temperature for more focused code responses
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1  # Reduce repetition
                }
            }
            
            # Make request
            response = requests.post(api_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                
                # Enhanced post-processing
                answer = self._post_process_response(generated_text, context)
                return answer
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"I apologize, but I encountered an API error: {response.status_code}"
            
        except Exception as e:
            logger.error(f"Error generating answer with Ollama: {e}")
            return f"I apologize, but I encountered an error while generating an answer: {str(e)}"
    
    async def _generate_with_hf_api(self, query: str, context: str, max_new_tokens: int = 200) -> str:
        """Generate answer using HuggingFace Inference API."""
        try:
            import requests
            
            # Get API key
            api_key = os.getenv("HUGGING_FACE_API_KEY")
            if not api_key:
                raise ValueError("HUGGING_FACE_API_KEY environment variable not set")
            
            # Extract model name (remove 'huggingface/' prefix)
            model_name = self.model_name.replace('huggingface/', '')
            
            # Prepare the prompt
            prompt = self._prepare_prompt(query, context)
            
            # API endpoint
            api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            
            # Headers
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.7,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            # Make request
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    # Clean up the response
                    answer = self._clean_response(generated_text)
                    return answer
                else:
                    return "I apologize, but I couldn't generate a proper response."
            else:
                logger.error(f"HuggingFace API error: {response.status_code} - {response.text}")
                return f"I apologize, but I encountered an API error: {response.status_code}"
            
        except Exception as e:
            logger.error(f"Error generating answer with HuggingFace API: {e}")
            return f"I apologize, but I encountered an error while generating an answer: {str(e)}"
    
    async def _generate_with_local(self, query: str, context: str, max_new_tokens: int = 200) -> str:
        """Generate answer using local model."""
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
        if not context.strip():
            # If no context, provide a direct response
            prompt_parts = [
                "You are a helpful coding assistant. Answer the following question:",
                "",
                "Question:",
                query,
                "",
                "Answer:"
            ]
        else:
            # With context, force the model to use it
            prompt_parts = [
                "You are a helpful coding assistant that answers questions about code using ONLY the provided context.",
                "IMPORTANT: Base your answer ONLY on the information in the context below. Do not generate generic responses.",
                "If the context doesn't contain enough information to answer the question, say so explicitly.",
                "",
                "Context:",
                context,
                "",
                "Question:",
                query,
                "",
                "Answer based on the context above:"
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
    
    def _post_process_response(self, response: str, context: str) -> str:
        """Enhanced post-processing with formatting and references."""
        import re
        
        # Clean up the response
        response = self._clean_response(response)
        
        # Extract file references from context
        file_references = self._extract_file_references(context)
        
        # Format code blocks if they're not properly formatted
        response = self._format_code_blocks(response)
        
        # Add references if we have file information
        if file_references:
            response = self._add_references(response, file_references)
        
        # Ensure proper markdown formatting
        response = self._format_markdown(response)
        
        return response
    
    def _extract_file_references(self, context: str) -> list:
        """Extract file references from context."""
        import re
        
        # Look for file patterns like `filename.py` or `path/to/file.py`
        file_pattern = r'`([^`]+\.(py|js|ts|java|cpp|c|h|go|rs|rb|php|md|txt|yml|yaml|json|xml|html|css|scss|sql))`'
        files = re.findall(file_pattern, context)
        
        # Also look for "In file" patterns
        in_file_pattern = r'In file `([^`]+)`'
        in_files = re.findall(in_file_pattern, context)
        
        # Combine and deduplicate
        all_files = [f[0] for f in files] + in_files
        return list(set(all_files))
    
    def _format_code_blocks(self, response: str) -> str:
        """Ensure code blocks are properly formatted."""
        import re
        
        # Look for code that should be in code blocks but isn't
        # Pattern for Python-like code (def, class, import, etc.)
        code_patterns = [
            r'(def \w+\([^)]*\):)',
            r'(class \w+:)',
            r'(import \w+)',
            r'(from \w+ import)',
            r'(return \w+)',
            r'(if __name__ == "__main__":)'
        ]
        
        for pattern in code_patterns:
            # Find code that's not already in code blocks
            matches = re.finditer(pattern, response)
            for match in matches:
                start, end = match.span()
                # Check if it's already in a code block
                before = response[:start]
                after = response[end:]
                
                # If not in code block, add one
                if not (before.rstrip().endswith('```') or after.lstrip().startswith('```')):
                    response = response[:start] + f'```python\n{match.group()}\n```' + response[end:]
                    break  # Only process first match to avoid infinite loops
        
        return response
    
    def _add_references(self, response: str, file_references: list) -> str:
        """Add file references to the response."""
        if not file_references:
            return response
        
        # Create references section
        references = "\n\n**References:**\n"
        for file_ref in file_references[:3]:  # Limit to 3 references
            references += f"- `{file_ref}`\n"
        
        # Add to response
        response += references
        
        return response
    
    def _format_markdown(self, response: str) -> str:
        """Ensure proper markdown formatting."""
        import re
        
        # Ensure proper line breaks
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Ensure code blocks have proper spacing
        response = re.sub(r'```(\w+)\n', r'```\1\n', response)
        response = re.sub(r'\n```', r'\n```\n', response)
        
        # Clean up extra whitespace
        response = re.sub(r' +', ' ', response)
        response = re.sub(r'\n +', '\n', response)
        
        return response.strip()
    
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
        if self.is_api_model:
            return await self._generate_with_api_temperature(query, context, temperature)
        else:
            if self.pipeline is None:
                await self.initialize()
            return await self._generate_with_local_temperature(query, context, temperature)
    
    async def _generate_with_api_temperature(self, query: str, context: str, temperature: float = 0.7) -> str:
        """Generate answer using OpenAI API with custom temperature."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            client = openai.OpenAI(api_key=api_key)
            model_name = self.model_name.replace('openai/', '')
            prompt = self._prepare_prompt(query, context)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant that answers questions about code and programming. Use the provided context to give accurate and helpful answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content
            return self._clean_response(generated_text)
            
        except Exception as e:
            logger.error(f"Error generating answer with API temperature {temperature}: {e}")
            return f"I apologize, but I encountered an error while generating an answer: {str(e)}"
    
    async def _generate_with_local_temperature(self, query: str, context: str, temperature: float = 0.7) -> str:
        """Generate answer using local model with custom temperature."""
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