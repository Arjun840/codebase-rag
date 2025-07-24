#!/usr/bin/env python3
"""
Test and compare different code-savvy generation models.

This script helps you select the best code-aware language model for your RAG system
based on your hardware capabilities and requirements.
"""

import asyncio
import sys
import time
import torch
from pathlib import Path
from typing import Dict, List, Any
import psutil
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codebase_rag.core.generator import Generator


def get_system_info() -> Dict[str, Any]:
    """Get system information for model selection."""
    info = {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if info['gpu_available']:
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info['gpu_name'] = torch.cuda.get_device_properties(0).name
    
    return info


def get_recommended_models(system_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get recommended models based on system capabilities."""
    models = []
    
    # High-end systems (16GB+ RAM, good GPU)
    if system_info['memory_gb'] >= 16 and system_info['gpu_available']:
        models.extend([
            {
                'name': 'bigcode/starcoder2-15b',
                'display_name': 'StarCoder2 15B',
                'description': 'Excellent code understanding, large model',
                'size_gb': 30,
                'recommended_for': 'High-end systems with good GPU',
                'type': 'code_completion'
            },
            {
                'name': 'codellama/CodeLlama-13b-hf',
                'display_name': 'Code Llama 13B',
                'description': 'Strong code generation and understanding',
                'size_gb': 26,
                'recommended_for': 'High-end systems with good GPU',
                'type': 'code_completion'
            },
            {
                'name': 'microsoft/DialoGPT-large',
                'display_name': 'DialoGPT Large',
                'description': 'Good conversational AI, not code-specific',
                'size_gb': 1.5,
                'recommended_for': 'General conversation',
                'type': 'conversational'
            }
        ])
    
    # Mid-range systems (8-16GB RAM, any GPU)
    if system_info['memory_gb'] >= 8:
        models.extend([
            {
                'name': 'codellama/CodeLlama-7b-hf',
                'display_name': 'Code Llama 7B',
                'description': 'Good code understanding, moderate size',
                'size_gb': 14,
                'recommended_for': 'Mid-range systems',
                'type': 'code_completion'
            },
            {
                'name': 'bigcode/starcoder2-7b',
                'display_name': 'StarCoder2 7B',
                'description': 'Good code understanding, smaller than 15B',
                'size_gb': 14,
                'recommended_for': 'Mid-range systems',
                'type': 'code_completion'
            },
            {
                'name': 'microsoft/DialoGPT-medium',
                'display_name': 'DialoGPT Medium',
                'description': 'Current default, general purpose',
                'size_gb': 0.8,
                'recommended_for': 'Lightweight systems',
                'type': 'conversational'
            }
        ])
    
    # Lightweight systems (4-8GB RAM)
    if system_info['memory_gb'] >= 4:
        models.extend([
            {
                'name': 'microsoft/DialoGPT-small',
                'display_name': 'DialoGPT Small',
                'description': 'Lightweight conversational model',
                'size_gb': 0.5,
                'recommended_for': 'Lightweight systems',
                'type': 'conversational'
            },
            {
                'name': 'distilgpt2',
                'display_name': 'DistilGPT-2',
                'description': 'Very lightweight, general purpose',
                'size_gb': 0.3,
                'recommended_for': 'Very lightweight systems',
                'type': 'general'
            }
        ])
    
    # API-based options (no local resources needed)
    models.extend([
        {
            'name': 'openai/gpt-3.5-turbo',
            'display_name': 'GPT-3.5 Turbo (API)',
            'description': 'Excellent code understanding via API',
            'size_gb': 0,
            'recommended_for': 'Any system with internet',
            'type': 'api',
            'requires_api_key': True
        },
        {
            'name': 'openai/gpt-4',
            'display_name': 'GPT-4 (API)',
            'description': 'Best code understanding via API',
            'size_gb': 0,
            'recommended_for': 'Any system with internet',
            'type': 'api',
            'requires_api_key': True
        }
    ])
    
    return models


async def test_model(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single generation model."""
    print(f"\n🔍 Testing {model_info['display_name']}...")
    print(f"   Description: {model_info['description']}")
    print(f"   Size: {model_info['size_gb']} GB")
    
    # Skip API models for local testing
    if model_info['type'] == 'api':
        print("   ⏭️ Skipping API model for local testing")
        return {
            'status': 'skipped',
            'reason': 'API model - requires separate testing'
        }
    
    try:
        start_time = time.time()
        
        # Initialize the generator
        generator = Generator(model_name=model_info['name'])
        await generator.initialize()
        
        # Test with a code-related query
        test_query = "How do I implement a binary search algorithm in Python?"
        test_context = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
"""
        
        # Generate answer
        answer = await generator.generate_answer(test_query, test_context, max_new_tokens=150)
        
        end_time = time.time()
        
        # Clean up
        await generator.cleanup()
        
        result = {
            'status': 'success',
            'time_taken': end_time - start_time,
            'answer_length': len(answer),
            'answer_preview': answer[:100] + "..." if len(answer) > 100 else answer
        }
        
        print(f"  ✅ Success!")
        print(f"  ⏱️ Time taken: {result['time_taken']:.2f}s")
        print(f"  📝 Answer length: {result['answer_length']} chars")
        print(f"  💬 Preview: {result['answer_preview']}")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Failed: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }


async def main():
    """Main function to test generation models."""
    print("🚀 Testing Code-Savvy Generation Models")
    print("=" * 60)
    
    # Get system information
    system_info = get_system_info()
    print(f"💻 System Information:")
    print(f"   CPU Cores: {system_info['cpu_count']}")
    print(f"   Memory: {system_info['memory_gb']:.1f} GB")
    print(f"   GPU Available: {system_info['gpu_available']}")
    if system_info['gpu_available']:
        print(f"   GPU: {system_info['gpu_name']}")
        print(f"   GPU Memory: {system_info['gpu_memory_gb']:.1f} GB")
    
    # Get recommended models
    models = get_recommended_models(system_info)
    
    print(f"\n📋 Recommended Models for Your System:")
    print("-" * 60)
    
    for i, model in enumerate(models, 1):
        print(f"{i:2d}. {model['display_name']}")
        print(f"    {model['description']}")
        print(f"    Size: {model['size_gb']} GB | Type: {model['type']}")
        if 'requires_api_key' in model:
            print(f"    ⚠️  Requires API key")
        print()
    
    # Test local models
    local_models = [m for m in models if m['type'] != 'api']
    
    print(f"🧪 Testing Local Models...")
    print("=" * 60)
    
    results = {}
    
    for model in local_models:
        results[model['display_name']] = await test_model(model)
    
    # Print summary
    print("\n📊 Test Summary")
    print("=" * 60)
    print(f"{'Model':<25} {'Status':<10} {'Time(s)':<8} {'Length':<8}")
    print("-" * 60)
    
    for model_name, result in results.items():
        if result['status'] == 'success':
            print(f"{model_name:<25} {'✅ OK':<10} {result['time_taken']:<8.2f} {result['answer_length']:<8}")
        elif result['status'] == 'skipped':
            print(f"{model_name:<25} {'⏭️ SKIP':<10} {'N/A':<8} {'N/A':<8}")
        else:
            print(f"{model_name:<25} {'❌ FAIL':<10} {'N/A':<8} {'N/A':<8}")
    
    # Recommendations
    print("\n🎯 Recommendations")
    print("=" * 60)
    
    successful_models = [name for name, result in results.items() if result['status'] == 'success']
    
    if successful_models:
        print("✅ Working models (recommended for use):")
        for model_name in successful_models:
            model = next(m for m in local_models if m['display_name'] == model_name)
            print(f"   • {model_name} - {model['description']}")
        
        # Find best code-aware model
        code_models = [m for m in local_models if m['type'] == 'code_completion' and m['display_name'] in successful_models]
        if code_models:
            best_code_model = code_models[0]  # First one is usually the best
            print(f"\n🏆 Best code-aware model: {best_code_model['display_name']}")
            print(f"   Set this in your .env file:")
            print(f"   GENERATION_MODEL={best_code_model['name']}")
    else:
        print("❌ No local models worked. Consider:")
        print("   • Using API models (GPT-3.5/GPT-4)")
        print("   • Upgrading your system resources")
        print("   • Checking your internet connection")
    
    # API recommendations
    api_models = [m for m in models if m['type'] == 'api']
    if api_models:
        print(f"\n🌐 API Models (excellent for code understanding):")
        for model in api_models:
            print(f"   • {model['display_name']} - {model['description']}")
            if 'requires_api_key' in model:
                print(f"     Requires OpenAI API key")
    
    print("\n✨ Testing completed!")
    print("\nNext steps:")
    print("1. Choose a working model from the recommendations above")
    print("2. Set GENERATION_MODEL in your .env file")
    print("3. Test the model with your RAG system")


if __name__ == "__main__":
    asyncio.run(main()) 