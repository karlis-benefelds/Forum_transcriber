# Class Transcriber

A professional Python tool for transcribing lecture audio using OpenAI's Whisper model, with Forum integration and comprehensive reporting capabilities.

## 🎯 Features

- **🎵 Audio Processing**: Handles various formats (MP4, MP3, WAV, etc.) with automatic optimization for Whisper
- **🤖 AI Transcription**: Uses Whisper 'medium' model for accurate speech-to-text with word-level timestamps  
- **🔗 Forum Integration**: Fetches class events, attendance, and metadata from Forum API
- **📊 Smart Reports**: Generates comprehensive PDF and CSV reports with speaker identification
- **🔒 Privacy Options**: Support for both named and anonymized student reports
- **⚡ Performance**: GPU acceleration and batch processing for faster transcription

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- `ffmpeg` (required for audio processing)

### Installation

**Option 1: Automated Setup (Recommended)**
```bash
# Clone the repository
git clone <repository_url>
cd class-transcriber

# Run the setup script
make setup
```

**Option 2: Manual Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run with command line arguments
python src/main.py \
  --curl "your_forum_curl_string" \
  --audio_path "/path/to/audio.mp4" \
  --privacy_mode "names"

# Or get help
python src/main.py --help
```

## 📋 Command Line Options

| Option | Description | Required | Default |
|--------|-------------|----------|---------|
| `--curl` | Forum cURL string for API authentication | ✅ | - |
| `--audio_path` | Path or URL to media file | ✅ | - |
| `--privacy_mode` | Privacy mode: `names`, `ids`, or `both` | ❌ | `names` |
| `--class_id` | Class ID (auto-detected from cURL if not provided) | ❌ | Auto-detect |
| `--user_terms` | Comma-separated custom terms to preserve | ❌ | None |

## 🛠️ Development

### Available Commands

```bash
make help        # Show all available commands
make install     # Install dependencies and setup environment
make test        # Run tests with coverage
make lint        # Run code linting
make format      # Format code with black
make clean       # Clean up temporary files
make dev         # Setup development environment
```

### Running Tests

```bash
make test
# or manually:
pytest tests/ -v --cov=src
```

### Project Structure

```
class-transcriber/
├── src/                    # Main source code
│   ├── main.py            # Entry point and CLI interface
│   ├── audio_preprocessor.py
│   ├── transcription_processor.py
│   ├── forum_data_fetcher.py
│   ├── report_generator.py
│   └── utils.py
├── tests/                 # Test suite
├── config/               # Configuration files
│   └── config.yaml
├── scripts/              # Setup and utility scripts
├── notebooks/            # Original Jupyter notebook
├── output/               # Generated reports (created automatically)
└── docs/                 # Documentation
```

## ⚙️ Configuration

The project uses `config/config.yaml` for configuration. Key settings include:

- **Audio processing**: Model size, segment length, sample rate
- **Output formats**: File naming, directory structure
- **Privacy settings**: Default modes, report inclusions
- **Performance**: GPU usage, mixed precision

## 📁 Output Files

Reports are generated in the `output/` directory:

- `session_{class_id}_transcript_names.pdf` - Full transcript with student names
- `session_{class_id}_transcript_ids.pdf` - Anonymized transcript
- `session_{class_id}_transcript_names.csv` - CSV format with names
- `session_{class_id}_transcript_ids.csv` - CSV format with IDs

## 🔧 Getting Forum cURL

1. Open your browser's Developer Tools (F12)
2. Navigate to the Forum class page
3. Go to Network tab and refresh
4. Find a request to the Forum API
5. Right-click → Copy → Copy as cURL

## ⚠️ Important Notes

- **Accuracy**: AI transcripts are not 100% accurate - always verify critical information
- **Privacy**: Be mindful of student privacy when sharing transcripts
- **Resources**: GPU acceleration recommended for faster processing
- **Dependencies**: Ensure ffmpeg is properly installed for audio processing

## 🔍 Troubleshooting

**Common Issues:**

1. **ffmpeg not found**: Install with `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Ubuntu)
2. **CUDA errors**: Ensure PyTorch CUDA version matches your system
3. **Forum API errors**: Verify your cURL string is recent and valid
4. **Memory issues**: Reduce segment length in config for large files

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request
