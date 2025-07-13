#!/bin/bash

# Codebase RAG Setup Script
# This script sets up the RAG system for codebase search

set -e

echo "🚀 Setting up Codebase RAG System"
echo "================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "❌ Python 3.9 or higher is required"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install the package
echo "🔧 Installing Codebase RAG..."
pip install -e .

# Install additional dependencies for better performance
echo "🚀 Installing additional dependencies..."
pip install -e ".[all]"

# Create configuration directory
echo "📁 Creating configuration directories..."
mkdir -p data/vector_db
mkdir -p logs

# Copy example configuration
echo "⚙️ Setting up configuration..."
cp env.example .env

# Check if everything is working
echo "✅ Testing installation..."
python3 -c "import codebase_rag; print('✅ Installation successful!')"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Edit .env file with your preferences"
echo "3. Index your codebase: codebase-rag index /path/to/your/codebase"
echo "4. Start the web interface: codebase-rag web"
echo ""
echo "For more information, see the README.md file" 