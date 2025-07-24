#!/usr/bin/env python3
"""
Cost calculator for GPT-3.5 Turbo API usage in RAG system.
"""

import tiktoken
import json

def count_tokens(text):
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def calculate_cost(input_text, output_text):
    """Calculate cost for input and output text."""
    input_tokens = count_tokens(input_text)
    output_tokens = count_tokens(output_text)
    
    # GPT-3.5 Turbo pricing
    input_cost = (input_tokens / 1000) * 0.0015
    output_cost = (output_tokens / 1000) * 0.002
    
    total_cost = input_cost + output_cost
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost
    }

def estimate_rag_costs():
    """Estimate costs for typical RAG usage scenarios."""
    
    scenarios = [
        {
            'name': 'Simple Code Question',
            'input': 'How do I implement a binary search in Python?',
            'context': 'def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1',
            'output': 'This is a binary search implementation in Python. It works by dividing the search interval in half each time...'
        },
        {
            'name': 'Codebase Query',
            'input': 'How does the authentication system work in this codebase?',
            'context': 'Here is the authentication module with login, register, and token validation functions...',
            'output': 'The authentication system consists of three main components: user registration, login, and token validation...'
        },
        {
            'name': 'Error Debugging',
            'input': 'I\'m getting "ImportError: No module named \'requests\'". How do I fix this?',
            'context': 'This error occurs when the requests library is not installed in your Python environment.',
            'output': 'To fix this ImportError, you need to install the requests library using pip...'
        },
        {
            'name': 'Code Explanation',
            'input': 'Explain this sorting algorithm',
            'context': 'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)',
            'output': 'This is a quicksort implementation in Python. Quicksort is a divide-and-conquer algorithm that works by selecting a pivot element...'
        }
    ]
    
    print("💰 GPT-3.5 Turbo Cost Calculator")
    print("=" * 50)
    
    total_monthly_cost = 0
    
    for scenario in scenarios:
        result = calculate_cost(scenario['input'] + '\n\nContext:\n' + scenario['context'], scenario['output'])
        
        print(f"\n📝 {scenario['name']}:")
        print(f"   Input tokens: {result['input_tokens']:,}")
        print(f"   Output tokens: {result['output_tokens']:,}")
        print(f"   Cost: ${result['total_cost']:.4f}")
        
        # Estimate monthly cost (assuming 25 questions of this type per month)
        monthly_cost = result['total_cost'] * 25
        total_monthly_cost += monthly_cost
        print(f"   Monthly (25x): ${monthly_cost:.2f}")
    
    print(f"\n📊 Monthly Total (100 questions): ${total_monthly_cost:.2f}")
    
    # Usage recommendations
    print(f"\n💡 Usage Recommendations:")
    print(f"   Light usage (30 questions/month): ${total_monthly_cost * 0.3:.2f}")
    print(f"   Moderate usage (100 questions/month): ${total_monthly_cost:.2f}")
    print(f"   Heavy usage (300 questions/month): ${total_monthly_cost * 3:.2f}")
    
    print(f"\n🎯 Cost Comparison:")
    print(f"   GPT-3.5 Turbo (100 questions): ${total_monthly_cost:.2f}")
    print(f"   GPT-4 (100 questions): ~${total_monthly_cost * 10:.2f}")
    print(f"   Local models: $0 (but requires hardware)")

if __name__ == "__main__":
    estimate_rag_costs() 