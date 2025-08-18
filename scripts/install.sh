#!/bin/bash

# Class Transcriber Installation Script
# This script sets up the development environment for the Class Transcriber project

set -e  # Exit on any error

echo "🚀 Setting up Class Transcriber development environment..."

# Check if Python 3.8+ is installed
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.8 or higher is required. Found: $python_version"
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

echo "✅ Python version: $(python3 --version)"

# Check if ffmpeg is installed
echo "📋 Checking for ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ Error: ffmpeg is required but not installed."
    echo "Please install ffmpeg:"
    echo "  - macOS: brew install ffmpeg"
    echo "  - Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg"
    echo "  - Windows: Download from https://ffmpeg.org/download.html"
    exit 1
fi

echo "✅ ffmpeg is installed: $(ffmpeg -version | head -n1)"

# Create virtual environment
echo "🔨 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv
echo "✅ Virtual environment created"

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Create output directory
echo "📁 Creating output directory..."
mkdir -p output
mkdir -p logs

# Create environment file template
echo "📝 Creating environment file template..."
cat > .env.template << EOF
# Class Transcriber Environment Variables
# Copy this file to .env and fill in your values

# Optional: Custom output directory
# OUTPUT_DIR=./output

# Optional: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# LOG_LEVEL=INFO

# Optional: Custom config file path
# CONFIG_FILE=./config/config.yaml
EOF

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Copy .env.template to .env and configure if needed:"
echo "   cp .env.template .env"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run the transcriber:"
echo "   python src/main.py --help"
echo ""
echo "📚 For detailed usage instructions, see README.md"