#!/usr/bin/env python3
"""
Test script for Ollama setup.
"""

import asyncio
import sys
import subprocess
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codebase_rag.core.generator import Generator


def check_ollama_service():
    """Check if Ollama service is running."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True
        else:
            return False
    except Exception:
        return False


async def test_ollama():
    """Test Ollama setup."""
    print("🧪 Testing Ollama Setup")
    print("=" * 50)
    
    # Check if Ollama service is running
    print("🔍 Checking Ollama service...")
    if not check_ollama_service():
        print("❌ Ollama service not running!")
        print("📝 Please start Ollama:")
        print("   brew services start ollama")
        return False
    
    print("✅ Ollama service is running")
    
    # Check if Code Llama model is available
    print("🔍 Checking for Code Llama model...")
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if 'codellama:7b' in result.stdout:
            print("✅ Code Llama 7B model found")
        else:
            print("❌ Code Llama 7B model not found")
            print("📥 Downloading Code Llama 7B...")
            subprocess.run(['ollama', 'pull', 'codellama:7b'], check=True)
            print("✅ Code Llama 7B downloaded")
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False
    
    try:
        # Initialize generator with Ollama model
        generator = Generator(model_name="codellama/CodeLlama-7b-hf")
        
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
    success = await test_ollama()
    
    if success:
        print(f"\n🎉 Ollama setup is working!")
        print(f"💡 You can now use your RAG system with:")
        print(f"   GENERATION_MODEL=codellama/CodeLlama-7b-hf")
        print(f"")
        print(f"🚀 Benefits:")
        print(f"   ✅ Free to use")
        print(f"   ✅ Excellent code understanding")
        print(f"   ✅ Runs locally")
        print(f"   ✅ No API limits")
    else:
        print(f"\n❌ Setup incomplete. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main()) 