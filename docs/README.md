# Sloposcope Documentation

Welcome to the comprehensive documentation for Sloposcope, an AI text quality analysis system that detects AI-generated content patterns across multiple dimensions.

## 📋 Documentation Structure

### 🚀 Quick Start

- **[README.md](../README.md)** - Main project overview, installation, and usage
- **[DEPLOYMENT.md](../DEPLOYMENT.md)** - Deployment guides and production setup

### 🔧 Technical Documentation

- **[DOCS.md](../DOCS.md)** - Technical implementation details and API reference
- **[examples/README.md](../examples/README.md)** - Example texts and testing guides

### 🏗️ Architecture

- **Core Features**: 11-dimensional analysis framework
- **Web Interface**: FastAPI application with responsive UI
- **CLI Tool**: Command-line interface for batch processing
- **REST API**: Programmatic access to analysis features

## 🎯 Key Features

- **Multi-dimensional Analysis**: 11 distinct quality metrics
- **Domain-specific Tuning**: Optimized for general, news, and Q&A content
- **Real-time Processing**: Fast analysis with detailed explanations
- **Production Ready**: Docker deployment with monitoring endpoints

## 📊 Analysis Dimensions

1. **Density** - Information content per word
2. **Repetition** - N-gram repetition patterns
3. **Templated** - Formulaic language detection
4. **Tone** - Jargon and phrasing analysis
5. **Coherence** - Entity continuity and flow
6. **Relevance** - Context appropriateness
7. **Factuality** - Accuracy assessment
8. **Subjectivity** - Bias detection
9. **Fluency** - Grammar and language patterns
10. **Complexity** - Readability measures
11. **Verbosity** - Wordiness analysis

## 🚀 Quick Commands

```bash
# Install and setup
make install

# Run web interface
make run

# Run tests
make test

# Analyze text
sloposcope analyze --text "Your text here" --explain
```

## 📈 Slop Score Levels

- **🟢 Clean** (≤ 0.50): High-quality, human-like text
- **🟡 Watch** (0.50 - 0.70): Some AI patterns detected
- **🟠 Sloppy** (0.70 - 0.85): Clear AI-generated characteristics
- **🔴 High-Slop** (> 0.85): Obvious AI-generated content

## 🤝 Getting Help

- **Issues**: [GitHub Issues](https://github.com/rgilks/sloposcope/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rgilks/sloposcope/discussions)
- **API Docs**: Available at `/docs` when running the web interface

## 📝 Contributing

We welcome contributions! Please see the main [README.md](../README.md) for contribution guidelines.

---

_This documentation is automatically maintained and reflects the current state of the Sloposcope project._
