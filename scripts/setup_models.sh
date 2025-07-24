#!/bin/bash

# Setup script for code-savvy models
# This script helps install dependencies and test models

set -e

echo "🚀 Setting up Code-Savvy Models for RAG System"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "✅ Created .env file. Please edit it to configure your models."
fi

echo ""
echo "✅ Setup completed!"
echo ""
echo "🎯 Next steps:"
echo "1. Edit .env file to configure your preferred models"
echo "2. Test your system: python3 scripts/test_generation_models.py"
echo "3. Demo models: python3 scripts/demo_generation_models.py"
echo ""
echo "💡 For API models, add your OPENAI_API_KEY to .env"
echo "💡 For local models, ensure you have sufficient RAM and optionally GPU"
echo ""
echo "📖 See docs/CODE_SAVVY_MODELS_GUIDE.md for detailed information" 