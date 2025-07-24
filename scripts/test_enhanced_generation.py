#!/usr/bin/env python3
"""
Test script to demonstrate enhanced generation and post-processing with Ollama.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codebase_rag.core.generator import Generator


async def test_enhanced_generation():
    """Test the enhanced generation and post-processing."""
    print("🧪 Testing Enhanced Generation & Post-Processing")
    print("=" * 60)
    
    # Test case 1: Code sorting question
    user_question = "How can I sort a list of custom objects by an attribute using this library?"
    
    retrieved_context = """1. In file `models.py`:
   ```python
   class Item:
       def __init__(self, name, value):
           self.name = name
           self.value = value
   ```

2. In file `utils/sorting.py`:
   ```python
   def sort_items_by_value(items: List[Item]) -> List[Item]:
       \"\"\"Sort a list of Item objects by their value attribute.\"\"\"
       return sorted(items, key=lambda item: item.value)
   ```

3. In file `examples/sort_demo.py`:
   ```python
   from models import Item
   from utils.sorting import sort_items_by_value
   
   items = [Item("A", 3), Item("B", 1), Item("C", 2)]
   sorted_items = sort_items_by_value(items)
   print([item.name for item in sorted_items])  # ['B', 'C', 'A']
   ```"""
    
    print(f"📝 User Question: {user_question}")
    print(f"📚 Retrieved Context (3 files):")
    print("   - models.py (Item class)")
    print("   - utils/sorting.py (sorting function)")
    print("   - examples/sort_demo.py (usage example)")
    print("-" * 60)
    
    # Initialize generator with enhanced settings
    generator = Generator(model_name="codellama/CodeLlama-7b-hf")
    
    # Generate answer with enhanced post-processing
    print(f"🤖 Generating Answer with Enhanced Post-Processing...")
    answer = await generator.generate_answer(user_question, retrieved_context, max_new_tokens=250)
    
    print(f"✅ Enhanced Answer:")
    print("=" * 60)
    print(answer)
    print("=" * 60)


async def test_error_debugging():
    """Test error debugging with enhanced generation."""
    print("\n🧪 Testing Error Debugging")
    print("=" * 60)
    
    user_question = "I'm getting 'AttributeError: 'Item' object has no attribute 'price''. How do I fix this?"
    
    retrieved_context = """1. In file `models.py`:
   ```python
   class Item:
       def __init__(self, name, value):
           self.name = name
           self.value = value
   ```

2. In file `utils/sorting.py`:
   ```python
   def sort_items_by_value(items: List[Item]) -> List[Item]:
       \"\"\"Sort a list of Item objects by their value attribute.\"\"\"
       return sorted(items, key=lambda item: item.value)
   
   def sort_items_by_price(items: List[Item]) -> List[Item]:
       \"\"\"Sort a list of Item objects by their price attribute.\"\"\"
       return sorted(items, key=lambda item: item.price)  # This will cause an error!
   ```

3. In file `main.py`:
   ```python
   from models import Item
   from utils.sorting import sort_items_by_price
   
   items = [Item("Apple", 1.99), Item("Banana", 0.99)]
   sorted_items = sort_items_by_price(items)  # This will fail!
   ```"""
    
    print(f"📝 User Question: {user_question}")
    print(f"📚 Retrieved Context (3 files with error):")
    print("   - models.py (Item class has 'value', not 'price')")
    print("   - utils/sorting.py (function tries to access 'price')")
    print("   - main.py (usage that will cause error)")
    print("-" * 60)
    
    generator = Generator(model_name="codellama/CodeLlama-7b-hf")
    
    print(f"🤖 Generating Error Solution...")
    answer = await generator.generate_answer(user_question, retrieved_context, max_new_tokens=300)
    
    print(f"✅ Error Solution:")
    print("=" * 60)
    print(answer)
    print("=" * 60)


async def test_code_generation():
    """Test code generation with enhanced post-processing."""
    print("\n🧪 Testing Code Generation")
    print("=" * 60)
    
    user_question = "Write a function to filter items by a minimum value threshold"
    
    retrieved_context = """1. In file `models.py`:
   ```python
   class Item:
       def __init__(self, name, value):
           self.name = name
           self.value = value
   ```

2. In file `utils/sorting.py`:
   ```python
   def sort_items_by_value(items: List[Item]) -> List[Item]:
       \"\"\"Sort a list of Item objects by their value attribute.\"\"\"
       return sorted(items, key=lambda item: item.value)
   ```

3. In file `utils/filtering.py`:
   ```python
   def filter_items_by_name(items: List[Item], name_pattern: str) -> List[Item]:
       \"\"\"Filter items by name pattern.\"\"\"
       return [item for item in items if name_pattern in item.name]
   ```"""
    
    print(f"📝 User Question: {user_question}")
    print(f"📚 Retrieved Context (3 files):")
    print("   - models.py (Item class)")
    print("   - utils/sorting.py (sorting function)")
    print("   - utils/filtering.py (existing filtering function)")
    print("-" * 60)
    
    generator = Generator(model_name="codellama/CodeLlama-7b-hf")
    
    print(f"🤖 Generating Code...")
    answer = await generator.generate_answer(user_question, retrieved_context, max_new_tokens=300)
    
    print(f"✅ Generated Code:")
    print("=" * 60)
    print(answer)
    print("=" * 60)


async def main():
    """Main function."""
    print("🚀 Enhanced Generation & Post-Processing Demo")
    print("=" * 80)
    print("Testing Ollama with enhanced post-processing features:")
    print("✅ Optimized generation parameters")
    print("✅ Automatic code block formatting")
    print("✅ File reference extraction")
    print("✅ Markdown formatting")
    print("✅ Error handling and debugging")
    print("=" * 80)
    
    # Test different scenarios
    await test_enhanced_generation()
    await test_error_debugging()
    await test_code_generation()
    
    print(f"\n🎉 Enhanced Generation Demo Completed!")
    print(f"💡 Key Improvements:")
    print(f"   ✅ Lower temperature (0.3) for more focused code responses")
    print(f"   ✅ Automatic file reference extraction and citation")
    print(f"   ✅ Code block formatting and syntax highlighting")
    print(f"   ✅ Enhanced markdown formatting")
    print(f"   ✅ Better error handling and debugging capabilities")


if __name__ == "__main__":
    asyncio.run(main()) 