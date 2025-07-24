#!/usr/bin/env python3
"""
Demo script for testing code-savvy generation models.

This script demonstrates how different models handle code-related queries
and helps you choose the best model for your needs.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codebase_rag.core.generator import Generator


async def test_code_understanding(model_name: str, display_name: str):
    """Test a model's code understanding capabilities."""
    print(f"\n🧪 Testing {display_name}")
    print("=" * 50)
    
    try:
        # Initialize generator
        generator = Generator(model_name=model_name)
        
        # Test cases
        test_cases = [
            {
                "query": "What does this Python function do?",
                "context": """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
""",
                "description": "Code explanation"
            },
            {
                "query": "How can I fix this error: 'list' object has no attribute 'append'",
                "context": "This error typically occurs when you try to use list methods on a non-list object.",
                "description": "Error debugging"
            },
            {
                "query": "Write a function to find the longest common subsequence between two strings",
                "context": "The longest common subsequence (LCS) problem is a classic computer science problem.",
                "description": "Code generation"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test_case['description']}")
            print(f"Query: {test_case['query']}")
            
            # Generate answer
            answer = await generator.generate_answer(
                test_case['query'], 
                test_case['context'], 
                max_new_tokens=150
            )
            
            print(f"Answer: {answer}")
            print("-" * 30)
        
        # Cleanup
        await generator.cleanup()
        print(f"✅ {display_name} test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing {display_name}: {str(e)}")


async def main():
    """Main demo function."""
    print("🚀 Code-Savvy Generation Models Demo")
    print("=" * 60)
    print("This demo will test different models' ability to understand and work with code.")
    print("Models will be tested on code explanation, error debugging, and code generation.")
    
    # Models to test (in order of recommendation)
    models_to_test = [
        ("codellama/CodeLlama-7b-hf", "Code Llama 7B (Recommended)"),
        ("microsoft/DialoGPT-medium", "DialoGPT Medium (Current Default)"),
        ("openai/gpt-3.5-turbo", "GPT-3.5 Turbo (API - requires key)"),
    ]
    
    print(f"\n📋 Models to test:")
    for i, (model_id, display_name) in enumerate(models_to_test, 1):
        print(f"{i}. {display_name}")
    
    print(f"\n💡 Tips:")
    print("- Code Llama models are specifically trained for code understanding")
    print("- API models require OPENAI_API_KEY environment variable")
    print("- Local models require sufficient RAM and optionally GPU")
    
    # Test each model
    for model_id, display_name in models_to_test:
        await test_code_understanding(model_id, display_name)
    
    print(f"\n✨ Demo completed!")
    print(f"\n🎯 Recommendations:")
    print("1. For best code understanding: Use Code Llama or StarCoder models")
    print("2. For convenience: Use GPT-3.5/GPT-4 via API")
    print("3. For lightweight systems: Use DialoGPT or DistilGPT-2")
    print(f"\nTo set your preferred model, update your .env file:")
    print("GENERATION_MODEL=your_chosen_model_name")


if __name__ == "__main__":
    asyncio.run(main()) 