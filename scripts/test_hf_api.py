#!/usr/bin/env python3
"""
Test script for HuggingFace Inference API setup.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codebase_rag.core.generator import Generator


async def test_hf_api():
    """Test HuggingFace API setup."""
    print("🧪 Testing HuggingFace Inference API")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv("HUGGING_FACE_API_KEY")
    if not api_key:
        print("❌ HUGGING_FACE_API_KEY not set!")
        print("📝 Please:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token")
        print("3. Add it to your .env file:")
        print("   HUGGING_FACE_API_KEY=your_token_here")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        # Initialize generator with HuggingFace API model
        generator = Generator(model_name="huggingface/microsoft/DialoGPT-medium")
        
        # Test with a simple code question
        test_query = "What does this Python function do?"
        test_context = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
        
        print(f"🔍 Testing with query: {test_query}")
        print(f"📝 Context: {test_context[:100]}...")
        
        # Generate answer
        answer = await generator.generate_answer(test_query, test_context, max_new_tokens=100)
        
        print(f"✅ Success!")
        print(f"💬 Answer: {answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


async def main():
    """Main function."""
    success = await test_hf_api()
    
    if success:
        print(f"\n🎉 HuggingFace API setup is working!")
        print(f"💡 You can now use your RAG system with:")
        print(f"   GENERATION_MODEL=huggingface/codellama/CodeLlama-7b-hf")
        print(f"   HUGGING_FACE_API_KEY=your_token_here")
    else:
        print(f"\n❌ Setup incomplete. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main()) 