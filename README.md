# Sloposcope - AI Text Analysis

A comprehensive tool for detecting AI-generated text patterns and measuring "slop" across multiple dimensions.

## Features

- **Multi-dimensional Analysis**: 11 different metrics including density, repetition, coherence, templatedness, and more
- **Configurable Scoring**: Domain-specific weighting and calibration
- **Real-time Web Interface**: Modern web UI with instant analysis
- **REST API**: Full API for integration with other applications
- **High Performance**: Optimized for production deployment

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the application
uvicorn app:app --reload
```

Visit http://localhost:8000 to use the web interface.

### Deployment Options

This project supports two deployment targets:

- **Fly.io**: Use `./deploy-fly.sh` for Fly.io deployment
- **Docker**: Use the provided Dockerfile for containerized deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## API Usage

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

## Analysis Metrics

The tool analyzes text across 11 dimensions:

1. **Density** - Information density and perplexity measures
2. **Relevance** - How well content matches prompt/references
3. **Coherence** - Entity continuity and topic flow
4. **Repetition** - N-gram repetition and compression
5. **Verbosity** - Wordiness and structural complexity
6. **Templated** - Templated phrases and boilerplate detection
7. **Tone** - Hedging, sycophancy, and tone analysis
8. **Subjectivity** - Bias and subjectivity detection
9. **Fluency** - Grammar and fluency assessment
10. **Factuality** - Factual accuracy proxy
11. **Complexity** - Lexical and syntactic complexity

## Slop Levels

- **Clean** (≤ 0.30): High-quality, human-like text
- **Watch** (0.30 - 0.55): Some AI patterns detected
- **Sloppy** (0.55 - 0.75): Clear AI-generated characteristics
- **High-Slop** (> 0.75): Obvious AI-generated content

## Domains

The analysis can be customized for different domains:

- **General**: General purpose text analysis
- **News**: News articles and journalism
- **Q&A**: Question and answer content

## Development

### Running Tests

```bash
make test
```

### Code Quality

```bash
make lint
make format
```

## Architecture

- **Backend**: FastAPI with Python
- **Frontend**: Vanilla HTML/CSS/JavaScript with Tailwind CSS
- **Deployment**: Fly.io with Docker
- **ML Models**: spaCy, transformers, scikit-learn

## License

Apache 2.0 License - see LICENSE file for details.
