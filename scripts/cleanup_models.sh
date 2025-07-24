#!/bin/bash

# Cleanup script for managing model storage space
# This script helps you clean up unused models and manage storage

set -e

echo "🧹 Model Storage Cleanup Script"
echo "==============================="

# Show current usage
echo "📊 Current Storage Usage:"
echo "Project directory: $(du -sh . | cut -f1)"
echo "Virtual environment: $(du -sh venv/ 2>/dev/null | cut -f1 || echo 'Not found')"
echo "HuggingFace cache: $(du -sh ~/.cache/huggingface/ 2>/dev/null | cut -f1 || echo 'Not found')"

echo ""
echo "🗂️  Large models in cache:"
du -sh ~/.cache/huggingface/hub/models--*/ 2>/dev/null | sort -hr | head -5

echo ""
echo "🎯 Space-Saving Options:"
echo "1. Switch to API model (GPT-3.5/GPT-4) - saves ~26GB"
echo "2. Use smaller model (DialoGPT) - saves ~24GB"
echo "3. Remove specific large models"
echo "4. Clean entire HuggingFace cache"

echo ""
read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo "🔄 Switching to API model..."
        if [ -f ".env" ]; then
            sed -i '' 's/GENERATION_MODEL=.*/GENERATION_MODEL=openai\/gpt-3.5-turbo/' .env
            echo "✅ Updated .env to use GPT-3.5 Turbo"
            echo "⚠️  Don't forget to add your OPENAI_API_KEY to .env"
        else
            echo "❌ .env file not found. Please create one from env.example"
        fi
        ;;
    2)
        echo "🔄 Switching to smaller model..."
        if [ -f ".env" ]; then
            sed -i '' 's/GENERATION_MODEL=.*/GENERATION_MODEL=microsoft\/DialoGPT-medium/' .env
            echo "✅ Updated .env to use DialoGPT Medium"
        else
            echo "❌ .env file not found. Please create one from env.example"
        fi
        ;;
    3)
        echo "🗑️  Removing large models..."
        echo "Available models:"
        du -sh ~/.cache/huggingface/hub/models--*/ 2>/dev/null | sort -hr | head -10
        
        read -p "Enter model name to remove (e.g., codellama--CodeLlama-7b-hf): " model_name
        if [ -d "~/.cache/huggingface/hub/models--$model_name" ]; then
            rm -rf ~/.cache/huggingface/hub/models--$model_name
            echo "✅ Removed model: $model_name"
        else
            echo "❌ Model not found: $model_name"
        fi
        ;;
    4)
        echo "🗑️  Cleaning entire HuggingFace cache..."
        read -p "Are you sure? This will remove ALL cached models (y/N): " confirm
        if [[ $confirm == [yY] ]]; then
            rm -rf ~/.cache/huggingface/
            echo "✅ Cleared HuggingFace cache"
        else
            echo "❌ Cancelled"
        fi
        ;;
    *)
        echo "❌ Invalid option"
        ;;
esac

echo ""
echo "📊 Updated storage usage:"
echo "HuggingFace cache: $(du -sh ~/.cache/huggingface/ 2>/dev/null | cut -f1 || echo 'Not found')"
echo ""
echo "💡 Tips:"
echo "- API models require internet connection and API key"
echo "- Smaller models may have reduced code understanding"
echo "- You can always re-download models when needed" 