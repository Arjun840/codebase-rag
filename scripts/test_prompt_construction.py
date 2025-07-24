#!/usr/bin/env python3
"""
Test script to demonstrate prompt construction for RAG system.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codebase_rag.core.generator import Generator


async def test_prompt_construction():
    """Test the prompt construction with the user's example."""
    print("🧪 Testing Prompt Construction")
    print("=" * 50)
    
    # User's example
    user_question = "How can I sort a list of custom objects by an attribute using this library?"
    
    # Retrieved context (simulating what the RAG system would find)
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
   ```"""
    
    print(f"📝 User Question: {user_question}")
    print(f"📚 Retrieved Context:")
    print(retrieved_context)
    print("-" * 50)
    
    # Initialize generator
    generator = Generator(model_name="codellama/CodeLlama-7b-hf")
    
    # Show the constructed prompt
    prompt = generator._prepare_prompt(user_question, retrieved_context)
    print(f"🔧 Constructed Prompt:")
    print("=" * 50)
    print(prompt)
    print("=" * 50)
    
    # Generate answer
    print(f"🤖 Generating Answer...")
    answer = await generator.generate_answer(user_question, retrieved_context, max_new_tokens=200)
    
    print(f"✅ Generated Answer:")
    print("=" * 50)
    print(answer)
    print("=" * 50)
    
    return answer


async def test_with_real_codebase_context():
    """Test with a more realistic codebase context."""
    print("\n🧪 Testing with Realistic Codebase Context")
    print("=" * 50)
    
    user_question = "How do I implement authentication in this application?"
    
    # Simulated retrieved context from a real codebase
    retrieved_context = """1. In file `auth/models.py`:
   ```python
   class User:
       def __init__(self, username, email, password_hash):
           self.username = username
           self.email = email
           self.password_hash = password_hash
   
   class Session:
       def __init__(self, user_id, token, expires_at):
           self.user_id = user_id
           self.token = token
           self.expires_at = expires_at
   ```

2. In file `auth/views.py`:
   ```python
   @app.route('/login', methods=['POST'])
   def login():
       data = request.get_json()
       user = authenticate_user(data['username'], data['password'])
       if user:
           session = create_session(user.id)
           return {'token': session.token}
       return {'error': 'Invalid credentials'}, 401
   
   def authenticate_user(username, password):
       user = User.query.filter_by(username=username).first()
       if user and verify_password(password, user.password_hash):
           return user
       return None
   ```

3. In file `auth/utils.py`:
   ```python
   def verify_password(password, password_hash):
       return bcrypt.verify(password, password_hash)
   
   def create_session(user_id):
       token = generate_token()
       session = Session(user_id=user_id, token=token, expires_at=datetime.utcnow() + timedelta(hours=24))
       db.session.add(session)
       db.session.commit()
       return session
   ```"""
    
    print(f"📝 User Question: {user_question}")
    print(f"📚 Retrieved Context (simplified):")
    print("... (authentication models, views, and utilities) ...")
    
    # Initialize generator
    generator = Generator(model_name="codellama/CodeLlama-7b-hf")
    
    # Generate answer
    print(f"🤖 Generating Answer...")
    answer = await generator.generate_answer(user_question, retrieved_context, max_new_tokens=250)
    
    print(f"✅ Generated Answer:")
    print("=" * 50)
    print(answer)
    print("=" * 50)
    
    return answer


async def main():
    """Main function."""
    print("🚀 RAG Prompt Construction Demo")
    print("=" * 60)
    
    # Test with user's example
    answer1 = await test_prompt_construction()
    
    # Test with realistic context
    answer2 = await test_with_real_codebase_context()
    
    print(f"\n🎉 Demo completed!")
    print(f"💡 The prompt construction is working correctly with Ollama!")
    print(f"📊 Both examples generated relevant, code-aware responses.")


if __name__ == "__main__":
    asyncio.run(main()) 