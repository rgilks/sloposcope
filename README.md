# Sloposcope - AI Slop Detection System

A production-ready tool for detecting AI-generated text patterns and measuring "slop" based on the latest research in AI text quality assessment. Implements the 7-dimensional slop detection framework from "Measuring AI 'SLOP' in Text" (Shaib et al., 2025).

## ✨ Features

- **🔬 Research-Based**: Implements 7 core slop dimensions from academic research
- **🧠 Advanced NLP**: Transformer-based analysis with semantic understanding
- **📊 Multi-dimensional Analysis**: Density, Repetition, Templated, Tone, Coherence, Relevance, Factuality, Subjectivity, Fluency, and Complexity
- **🎯 Natural Writing Protection**: Prevents false positives on human-written content
- **⚡ High Performance**: Optimized for production with fast processing
- **🌐 Modern Web Interface**: Real-time analysis with beautiful, responsive UI
- **🔌 REST API**: Full API for seamless integration
- **☁️ Cloud Ready**: Docker deployment with Fly.io support
- **🔒 Privacy-First**: All processing happens locally, no external API calls

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- `uv` package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/rgilks/sloposcope.git
cd sloposcope

# One-command setup (installs everything)
make install
```

That's it! The setup process will:

- Install all dependencies with `uv`
- Download the required spaCy model
- Set up pre-commit hooks for automatic code quality checks
- Verify everything is working correctly

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

- **Automatic formatting** with Black and Ruff
- **Linting** to catch issues early
- **Unit tests** run on every commit
- **Comprehensive tests** run by pre-commit.ci on every push

The hooks are automatically installed during `make install`. If you need to set them up manually:

```bash
make setup-hooks
```

### Quick Test

```bash
# Test that everything is working
uv run sloplint analyze --text "This is a test" --explain
```

### Running Comprehensive Tests

For more thorough testing with 100+ diverse text samples:

```bash
# Run comprehensive test suite
uv run python tests/test_comprehensive_slop_detection.py
```

This test suite includes:

- Performance benchmarking (processing time per text)
- Accuracy testing across different content categories
- Slop score distribution analysis

### Command Line Usage

```bash
# Analyze text directly
uv run sloplint analyze --text "Your text here" --explain

# Analyze a file
uv run sloplint analyze document.txt --explain

# Get help
uv run sloplint --help
```

> **Note**: You may see a harmless warning about `loss_type=None` from the transformers library. This can be ignored.

### Web Interface

```bash
# Start the development server
uv run uvicorn app:app --reload

# Or use the Makefile
make run
```

Visit http://localhost:8000 to use the web interface.

## 🔍 Research-Based Analysis

The tool implements a comprehensive 11-dimensional analysis framework for detecting AI-generated text patterns:

### Core Dimensions

1. **Density** - Information content per word, detects verbose low-value content
2. **Repetition** - N-gram repetition and compression patterns
3. **Templated** - Formulaic and boilerplate language detection
4. **Tone** - Jargon and awkward phrasing detection
5. **Coherence** - Entity continuity and topic flow analysis
6. **Relevance** - Appropriateness to context/task
7. **Factuality** - Accuracy and truthfulness measures
8. **Subjectivity** - Bias and subjective language detection
9. **Fluency** - Grammar and natural language patterns
10. **Complexity** - Text complexity and readability measures
11. **Verbosity** - Wordiness and structural complexity

## 📈 Slop Levels

- **🟢 Clean** (≤ 0.50): High-quality, human-like text
- **🟡 Watch** (0.50 - 0.70): Some AI patterns detected
- **🟠 Sloppy** (0.70 - 0.85): Clear AI-generated characteristics
- **🔴 High-Slop** (> 0.85): Obvious AI-generated content

## 🎯 Domains

The analysis can be customized for different content types:

- **General**: General purpose text analysis
- **News**: News articles and journalism
- **Q&A**: Question and answer content

## 📡 API Usage

### Analyze Text

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here",
    "domain": "general",
    "explain": true,
    "spans": true
  }'
```

### Health Check

```bash
curl "http://localhost:8000/health"
```

### Get Metrics Info

```bash
curl "http://localhost:8000/metrics"
```

## 🌐 Deployment Options

### Fly.io (Recommended)

```bash
# Deploy to Fly.io
./deploy-fly.sh

# Your app will be available at https://sloposcope.fly.dev
```

### Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:8000
```

## 🛠️ Development

### Quick Commands

```bash
# Run tests
make test

# Lint code
make lint

# Format code
make format

# Run development server
make run

# Clean up
make clean
```

### Development Setup

```bash
# One-command setup (if not already done)
make install

# Start developing!
make run
```

## 🏗️ Architecture

- **Main Library**: `sloplint/` - Core feature extraction and analysis modules
- **CLI Interface**: `sloplint/cli.py` - Command-line interface
- **Web App**: `app.py` - FastAPI web application
- **NLP Pipeline**: `sloplint/nlp/` - spaCy with transformer models + sentence-transformers
- **Feature Extractors**: `sloplint/features/` - Individual slop dimension analyzers
- **Deployment**: Docker with Fly.io support
- **Testing**: pytest with comprehensive test coverage

## 📊 Performance

- **Speed**: <1s processing time for 1k words
- **Memory**: <400MB peak memory consumption
- **Accuracy**: Research-based implementation with high precision
- **Coverage**: >95% test coverage for core functionality

## 📚 Research Foundation

This project implements multiple research-based approaches for detecting AI-generated text quality issues, including patterns identified in recent academic literature on LLM text analysis. The implementation focuses on the most effective patterns for detecting verbose, templated, and low-quality AI-generated content.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [spaCy](https://spacy.io/) and [sentence-transformers](https://www.sbert.net/)
- Deployed on [Fly.io](https://fly.io/)
- Inspired by ongoing research in AI text detection and quality assessment
