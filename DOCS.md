# Sloposcope Technical Documentation

## Architecture Overview

Sloposcope is an AI text quality analysis system that implements a comprehensive 11-dimensional framework for detecting AI-generated content patterns. The system provides both CLI and web interfaces for analyzing text across multiple quality dimensions.

## Implementation Details

### Architecture Components

- **Core Engine** (`sloplint/`): Main analysis orchestration and feature extraction
- **Feature Extractors** (`sloplint/features/`): Individual dimension analyzers
- **NLP Pipeline** (`sloplint/nlp/`): Text processing with spaCy and transformers
- **Web Interface** (`app.py`): FastAPI application with HTML frontend
- **CLI Interface** (`sloplint/cli.py`): Command-line analysis tools

### Analysis Dimensions

The system implements 11 distinct quality dimensions:

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

## Technical Usage

### CLI Analysis

```bash
# Basic analysis with the updated command name
sloposcope analyze --text "This is a test sentence."

# Detailed analysis with explanations
sloposcope analyze document.txt --explain --spans

# Domain-specific analysis
sloposcope analyze article.txt --domain news --explain

# JSON output
sloposcope analyze input.txt --json output.json
```

### Web Interface

1. Start the server: `uv run uvicorn app:app --reload`
2. Open http://localhost:8000
3. Paste your text and click "Analyze Text"
4. View detailed results with visual indicators

### REST API

```python
import requests

response = requests.post("http://localhost:8000/analyze", json={
    "text": "Your text here",
    "domain": "general",  # general, news, qa
    "explain": True,
    "spans": False
})

result = response.json()
print(f"Slop Score: {result['slop_score']:.3f} ({result['level']})")
```

### Health Check

```bash
curl "http://localhost:8000/health"
```

### Metrics Information

```bash
curl "http://localhost:8000/metrics"
```

## Understanding Results

### Slop Score (0.0 - 1.0)

- **≤ 0.50**: Clean (high-quality text)
- **0.50-0.70**: Watch (some AI patterns)
- **0.70-0.85**: Sloppy (clear AI characteristics)
- **> 0.85**: High-Slop (obvious AI generation)

### Confidence (0.0 - 1.0)

- **1.0**: High confidence in analysis
- **0.5-0.9**: Moderate confidence
- **<0.5**: Low confidence (may need more text)

### Individual Metrics

Each dimension provides specific normalized scores (0.0-1.0):

- **Density**: Information content and perplexity measures
- **Repetition**: N-gram repetition and compression patterns
- **Templated**: Formulaic and boilerplate language detection
- **Tone**: Jargon and awkward phrasing detection
- **Coherence**: Entity continuity and topic flow analysis
- **Relevance**: Appropriateness to context/task
- **Factuality**: Accuracy and truthfulness measures
- **Subjectivity**: Bias and subjective language detection
- **Fluency**: Grammar and natural language patterns
- **Complexity**: Text complexity and readability measures
- **Verbosity**: Wordiness and structural complexity

## Configuration

### Domains

- **general**: Default, balanced analysis
- **news**: Optimized for journalistic content
- **qa**: Focused on question-answer content

### Customization

The system can be customized through:

- Domain-specific weights
- Threshold adjustments
- Feature selection
- Calibration data

## Troubleshooting

### Common Issues

1. **Low confidence scores**: Try longer text samples
2. **Inconsistent results**: Check domain selection
3. **Performance issues**: Ensure sufficient memory
4. **Model errors**: Verify spaCy model installation

### Getting Help

- Check the GitHub issues
- Review the API documentation at `/docs`
- Test with the provided sample texts

## Development

### Architecture

```
sloplint/
├── cli.py              # Command-line interface
├── feature_extractor.py # Main analysis orchestrator
├── combine.py          # Score normalization and combination
├── features/           # Individual dimension analyzers
├── nlp/               # NLP pipeline and processing
└── spans.py           # Character-level span detection
```

### Adding New Features

1. Create feature extractor in `features/`
2. Add dimension mapping in `combine.py`
3. Update CLI and web interface
4. Add tests and documentation

### Testing

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-coverage
```

## Research Background

This implementation is based on academic research that identified the most effective patterns for detecting AI-generated text quality issues. The system prioritizes:

- **Precision over recall**: Minimize false positives
- **Interpretability**: Clear explanations for each metric
- **Scalability**: Efficient processing for production use
- **Flexibility**: Adaptable to different content types

## Performance Considerations

- **Memory**: ~400MB peak usage
- **Speed**: <1s for 1000 words
- **Accuracy**: Research-validated thresholds
- **Scalability**: Batch processing support

## Future Enhancements

- Machine learning integration
- Real-time span detection
- Multi-language support
- Advanced calibration options
- Performance optimizations
