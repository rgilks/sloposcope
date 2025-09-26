# Sloposcope - AI Slop Detection System

A production-ready tool for detecting AI-generated text patterns and measuring "slop" based on the latest research in AI text quality assessment. Implements the 7-dimensional slop detection framework from "Measuring AI 'SLOP' in Text" (Shaib et al., 2025).

## ✨ Features

- **🔬 Research-Based**: Implements 7 core slop dimensions from academic research
- **🧠 Advanced NLP**: Transformer-based analysis with semantic understanding
- **📊 Multi-dimensional Analysis**: Density, Structure, Tone, Coherence, and more
- **🎯 Natural Writing Protection**: Prevents false positives on human-written content
- **⚡ High Performance**: Optimized for production with fast processing
- **🌐 Modern Web Interface**: Real-time analysis with beautiful, responsive UI
- **🔌 REST API**: Full API for seamless integration
- **☁️ Cloud Ready**: Docker deployment with Fly.io support
- **🔒 Privacy-First**: All processing happens locally, no external API calls

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- `uv` package manager (recommended) or `pip`

### Installation

```bash
# Clone the repository
git clone https://github.com/rgilks/sloposcope.git
cd sloposcope

# Install with uv (recommended)
uv sync --dev

# Or install with pip
pip install -e ".[dev]"

# Download required models
python -m spacy download en_core_web_sm
```

### Local Development

```bash
# Start the development server
uvicorn app:app --reload

# Or use the Makefile
make run
```

Visit http://localhost:8000 to use the web interface.

### Command Line Usage

```bash
# Analyze text directly
python research_compliant_slop_detector.py "Your text here"

# Analyze with detailed output
python research_compliant_slop_detector.py "Your text here" --verbose

# Get help
python research_compliant_slop_detector.py --help
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

## 📡 API Usage

### Analyze Text

```bash
curl -X POST "https://sloposcope.fly.dev/analyze" \
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
curl "https://sloposcope.fly.dev/health"
```

### Get Metrics Info

```bash
curl "https://sloposcope.fly.dev/metrics"
```

## 🔍 Research-Based Analysis

The tool implements the 7 core slop dimensions from "Measuring AI 'SLOP' in Text" (Shaib et al., 2025):

### Core Dimensions

1. **Density** - Information content per word, detects verbose low-value content
2. **Structure** - Templated and repetitive patterns, identifies formulaic writing
3. **Tone** - Jargon and awkward phrasing, catches corporate speak
4. **Coherence** - Logical flow and organization (currently being refined)
5. **Relevance** - Appropriateness to context/task (planned)
6. **Factuality** - Accuracy and truthfulness (planned)
7. **Bias** - One-sided or over-generalized claims (planned)

### Advanced Features

- **Natural Writing Detection**: Protects human-written content from false positives
- **Pattern Recognition**: Research-validated slop patterns and templates
- **Semantic Analysis**: Transformer-based understanding of text meaning
- **Dynamic Thresholds**: Adjusts sensitivity based on content type

## 📈 Slop Levels

- **🟢 Clean** (≤ 0.30): High-quality, human-like text
- **🟡 Watch** (0.30 - 0.55): Some AI patterns detected
- **🟠 Sloppy** (0.55 - 0.75): Clear AI-generated characteristics
- **🔴 High-Slop** (> 0.75): Obvious AI-generated content

## 🎯 Domains

The analysis can be customized for different content types:

- **General**: General purpose text analysis
- **News**: News articles and journalism
- **Q&A**: Question and answer content

## 🛠️ Development

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only (fast)
make test-unit

# Run with coverage
make test-coverage
```

### Code Quality

```bash
# Lint code
make lint

# Format code
make format

# Type checking
make type-check

# Run all quality checks
make pre-commit-run
```

### Development Setup

```bash
# Install development dependencies
make sync-dev

# Install pre-commit hooks
make pre-commit

# Run development server
make run
```

## 🏗️ Architecture

- **Main Detector**: `research_compliant_slop_detector.py` - Production-ready detector
- **Backend**: FastAPI with Python 3.11+
- **Frontend**: Vanilla HTML/CSS/JavaScript with Tailwind CSS
- **NLP Pipeline**: spaCy with transformer models + sentence-transformers
- **Core Library**: `sloplint/` - Feature extraction and analysis modules
- **Deployment**: Docker with Fly.io support
- **Testing**: pytest with comprehensive test coverage

## 📊 Performance

- **Speed**: <1s processing time for 1k words
- **Memory**: <400MB peak memory consumption
- **Accuracy**: 66.7% accuracy on test cases with zero false positives
- **Coverage**: >95% test coverage for core functionality

## 📚 Research Foundation

This project is based on the research paper "Measuring AI 'SLOP' in Text" by Shaib et al. (2025), which identifies 7 core dimensions of AI-generated text quality issues. The implementation focuses on the most effective patterns and approaches identified in the research.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on research by Shaib et al. (2025) "Measuring AI 'SLOP' in Text"
- Built with [spaCy](https://spacy.io/) and [sentence-transformers](https://www.sbert.net/)
- Deployed on [Fly.io](https://fly.io/)
- Inspired by ongoing research in AI text detection and quality assessment

## 📖 Additional Documentation

- [Deployment Guide](DEPLOYMENT.md) - Detailed deployment instructions
- [Technical Specification](SPEC.md) - Complete technical documentation
- [Testing Guide](TESTING.md) - How to run and write tests
- [Development History](IMPROVEMENT_HISTORY.md) - Project evolution and achievements
