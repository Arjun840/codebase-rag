#!/bin/bash

# Setup script for free API options
echo "🆓 Free API Setup Options"
echo "========================"

echo ""
echo "🎯 Available Free Options:"
echo "1. HuggingFace Inference API (30K requests/month free)"
echo "2. Ollama (local, completely free)"
echo "3. Keep current setup with smaller models"
echo "4. Show cost comparison"

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "🔧 Setting up HuggingFace Inference API..."
        echo ""
        echo "📝 Steps:"
        echo "1. Go to https://huggingface.co/settings/tokens"
        echo "2. Create a new token"
        echo "3. Copy the token"
        echo ""
        read -p "Enter your HuggingFace API token: " hf_token
        
        if [ -f ".env" ]; then
            sed -i '' 's/GENERATION_MODEL=.*/GENERATION_MODEL=huggingface\/codellama\/CodeLlama-7b-hf/' .env
            echo "HUGGING_FACE_API_KEY=$hf_token" >> .env
            echo "✅ Updated .env for HuggingFace API"
        else
            echo "GENERATION_MODEL=huggingface/codellama/CodeLlama-7b-hf" > .env
            echo "HUGGING_FACE_API_KEY=$hf_token" >> .env
            echo "✅ Created .env for HuggingFace API"
        fi
        
        echo ""
        echo "💡 HuggingFace API Benefits:"
        echo "- 30,000 free requests per month"
        echo "- Access to Code Llama and other code models"
        echo "- No local storage needed"
        echo ""
        echo "⚠️  Limitations:"
        echo "- Slower than OpenAI"
        echo "- Rate limits apply"
        ;;
    2)
        echo "🔧 Setting up Ollama (local free option)..."
        echo ""
        echo "📝 Installing Ollama..."
        
        if command -v ollama &> /dev/null; then
            echo "✅ Ollama already installed"
        else
            echo "📥 Installing Ollama..."
            curl -fsSL https://ollama.ai/install.sh | sh
        fi
        
        echo ""
        echo "📥 Downloading Code Llama 7B model..."
        ollama pull codellama:7b
        
        if [ -f ".env" ]; then
            sed -i '' 's/GENERATION_MODEL=.*/GENERATION_MODEL=codellama/CodeLlama-7b-hf/' .env
            echo "✅ Updated .env for Ollama"
        else
            echo "GENERATION_MODEL=codellama/CodeLlama-7b-hf" > .env
            echo "✅ Created .env for Ollama"
        fi
        
        echo ""
        echo "💡 Ollama Benefits:"
        echo "- Completely free"
        echo "- No API limits"
        echo "- Runs locally"
        echo ""
        echo "⚠️  Requirements:"
        echo "- ~8GB RAM available"
        echo "- Slower than API options"
        ;;
    3)
        echo "🔧 Keeping current setup with smaller models..."
        
        # Clean up large models
        echo "🗑️  Removing large models to save space..."
        rm -rf ~/.cache/huggingface/hub/models--codellama--CodeLlama-7b-hf/ 2>/dev/null
        rm -rf ~/.cache/huggingface/hub/models--bigcode--starcoder2-7b/ 2>/dev/null
        
        if [ -f ".env" ]; then
            sed -i '' 's/GENERATION_MODEL=.*/GENERATION_MODEL=microsoft\/DialoGPT-medium/' .env
            echo "✅ Updated .env to use DialoGPT Medium"
        else
            echo "GENERATION_MODEL=microsoft/DialoGPT-medium" > .env
            echo "✅ Created .env with DialoGPT Medium"
        fi
        
        echo ""
        echo "💡 Current Setup Benefits:"
        echo "- Completely free"
        echo "- No API needed"
        echo "- Works offline"
        echo ""
        echo "⚠️  Limitations:"
        echo "- Limited code understanding"
        echo "- Still requires some storage"
        ;;
    4)
        echo "💰 Cost Comparison:"
        echo ""
        echo "| Option | Monthly Cost | Setup Cost | Code Understanding |"
        echo "|--------|--------------|------------|-------------------|"
        echo "| HuggingFace API | $0 | $0 | ⭐⭐⭐⭐ |"
        echo "| Ollama | $0 | $0 | ⭐⭐⭐⭐ |"
        echo "| DialoGPT | $0 | $0 | ⭐⭐ |"
        echo "| GPT-3.5 Turbo | $2-8 | $0 | ⭐⭐⭐⭐⭐ |"
        echo "| GPT-4 | $15-50 | $0 | ⭐⭐⭐⭐⭐ |"
        echo ""
        echo "💡 Recommendation:"
        echo "- Best free option: HuggingFace API"
        echo "- Best local option: Ollama"
        echo "- Best overall: GPT-3.5 Turbo ($2-8/month)"
        ;;
    *)
        echo "❌ Invalid option"
        ;;
esac

echo ""
echo "✅ Setup completed!"
echo "📖 See docs/CODE_SAVVY_MODELS_GUIDE.md for more details" 