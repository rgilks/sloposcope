# Testing Guide

This document describes how to run tests for the Sloposcope AI Text Analysis project.

## 🧪 Test Types

### Unit Tests

- **Location**: `test_unit_enhanced.py`
- **Purpose**: Test core functionality without external dependencies
- **Speed**: Fast (< 30 seconds)
- **Dependencies**: None
- **Coverage**: Core feature extraction, NLP pipeline, and CLI functionality

### Integration Tests

- **Location**: `tests/test_aws_worker.py`
- **Purpose**: Test AWS integration components
- **Speed**: Medium (30-60 seconds)
- **Dependencies**: AWS services (can be mocked)
- **Coverage**: SQS handling, S3 operations, CloudWatch metrics

## 🚀 Quick Start

### Prerequisites

Make sure you have `uv` installed:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Run All Tests

```bash
make test
```

### Run Unit Tests Only

```bash
make test-unit
```

### Run Tests with Coverage

```bash
make test-coverage
```

## 🔧 Manual Testing

### Test Enhanced Features

```bash
# Test the enhanced NLP pipeline directly
python test_enhanced_direct.py

# Test specific functionality
python -c "
from sloplint.nlp.pipeline import NLPPipeline
pipeline = NLPPipeline(use_transformer=True)
result = pipeline.process('This is a test sentence.')
print(f'Model: {result[\"model_name\"]}')
print(f'Has transformer: {result[\"has_transformer\"]}')
print(f'Sentences: {len(result[\"sentences\"])}')
"
```

### Test CLI Functionality

```bash
# Test CLI with sample text
echo "This is a test document with some content." | sloplint --domain general --json output.json

# Test different domains
sloplint sample.txt --domain news --explain
sloplint sample.txt --domain qa --spans
```

## 🔄 Pre-commit Hooks

The project includes pre-commit hooks that automatically run quality checks:

### Install Pre-commit Hooks

```bash
make pre-commit
```

### Run Pre-commit Hooks Manually

```bash
make pre-commit-run
```

### Development Setup with uv

```bash
# Sync dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

### Hook Behavior

- **Unit tests**: Run on every commit (fast)
- **Linting**: Run on every commit (`ruff`, `black`)
- **Type checking**: Run on every commit (`mypy`)
- **Formatting**: Run on every commit (`trailing-whitespace`, `end-of-file-fixer`)
- **YAML/TOML validation**: Run on every commit

## ⚙️ Test Configuration

### Pytest Configuration

- **Main config**: `pyproject.toml`
- **Test discovery**: `tests/` directory and root test files
- **Markers**: `slow`, `integration`, `unit`

### Environment Variables

Tests automatically set these environment variables when needed:

```bash
# For AWS tests (when not mocked)
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
AWS_DEFAULT_REGION=us-east-1
```

## 📁 Test Structure

### Test Files

- `test_unit_enhanced.py`: Comprehensive unit tests for enhanced features
- `test_enhanced_direct.py`: Direct testing of NLP pipeline improvements
- `tests/test_cli.py`: CLI functionality tests
- `tests/test_aws_worker.py`: AWS integration tests
- `tests/test_comprehensive_slop_detection.py`: End-to-end slop detection tests

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Performance Tests**: Speed and memory usage validation
- **Accuracy Tests**: Validation against known datasets

## 🐳 Docker Testing

### Test with Docker Compose

```bash
make docker-test
```

### Clean Up Docker

```bash
make docker-clean
```

## 🔄 Continuous Integration

The CI pipeline runs:

1. **Linting** (`ruff`, `black`)
2. **Type checking** (`mypy`)
3. **Unit tests** (fast, always run)
4. **Integration tests** (when relevant files change)
5. **Coverage reporting**

## 🐛 Troubleshooting

### Tests Failing

```bash
# Run tests with verbose output
make test-unit -v

# Run specific test file
python -m pytest test_unit_enhanced.py -v

# Run specific test
python -m pytest test_unit_enhanced.py::test_enhanced_nlp_pipeline -v
```

### Pre-commit Hooks Failing

```bash
# Update pre-commit hooks
pre-commit autoupdate

# Run specific hook
pre-commit run pytest-unit --all-files

# Run all hooks
pre-commit run --all-files
```

### Model Loading Issues

```bash
# Check if spaCy models are installed
python -c "import spacy; print(spacy.util.get_package_path('en_core_web_trf'))"

# Download missing models
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm
```

### Memory Issues

```bash
# Check memory usage during tests
python -c "
import psutil
print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB')
"

# Run tests with memory profiling
python -m pytest test_unit_enhanced.py --profile
```

## 📊 Performance

### Test Execution Times

- **Unit tests**: ~5-15 seconds
- **Integration tests**: ~30-60 seconds
- **Full test suite**: ~60-120 seconds
- **Pre-commit hooks**: ~10-30 seconds

### Optimization Tips

- Use `make test-unit` for quick feedback during development
- Use `make test-coverage` for comprehensive testing
- Pre-commit hooks are optimized to run only relevant tests
- Use `pytest -x` to stop on first failure

## 📈 Coverage

Test coverage reports are generated in:

- **Terminal output**: Shows coverage summary
- **HTML**: `htmlcov/index.html` (detailed coverage report)
- **XML**: `coverage.xml` (for CI integration)

### Current Coverage Targets

- **Overall**: >90%
- **Core functionality**: >95%
- **Feature extractors**: >90%
- **NLP pipeline**: >95%
- **CLI components**: >85%

### Coverage Commands

```bash
# Generate coverage report
make test-coverage

# View HTML coverage report
open htmlcov/index.html

# Check coverage for specific module
python -m pytest sloplint/features/ --cov=sloplint.features --cov-report=html
```

## 🎯 Test Data

### Sample Texts

The project includes various sample texts for testing:

- **Clean text**: High-quality human-written content
- **AI-generated text**: Various AI slop patterns
- **Edge cases**: Empty text, very short text, very long text
- **Domain-specific**: News, Q&A, general content

### Test Datasets

- **Validation set**: Manually annotated examples
- **Benchmark set**: Standard evaluation texts
- **Adversarial set**: Challenging edge cases

## 🔍 Debugging Tests

### Verbose Output

```bash
# Run with maximum verbosity
python -m pytest test_unit_enhanced.py -vvv

# Show print statements
python -m pytest test_unit_enhanced.py -s

# Show local variables on failure
python -m pytest test_unit_enhanced.py -l
```

### Debugging Specific Issues

```bash
# Debug NLP pipeline
python -c "
from sloplint.nlp.pipeline import NLPPipeline
pipeline = NLPPipeline(use_transformer=True)
result = pipeline.process('Test text')
print('Pipeline result:', result)
"

# Debug feature extraction
python -c "
from sloplint.feature_extractor import FeatureExtractor
extractor = FeatureExtractor()
result = extractor.extract_features('Test text', domain='general')
print('Features:', result)
"
```

## 📚 Best Practices

### Writing Tests

1. **Test isolation**: Each test should be independent
2. **Clear naming**: Use descriptive test names
3. **Assertions**: Use specific assertions, not just `assert True`
4. **Fixtures**: Use pytest fixtures for common setup
5. **Mocking**: Mock external dependencies

### Test Organization

1. **Group related tests**: Use test classes for related functionality
2. **Use markers**: Mark tests as `slow`, `integration`, etc.
3. **Parametrize**: Use `@pytest.mark.parametrize` for multiple inputs
4. **Documentation**: Add docstrings to test functions

## 🚀 Future Improvements

### Planned Enhancements

- **Property-based testing**: Using `hypothesis` for property-based tests
- **Performance benchmarking**: Automated performance regression testing
- **Visual testing**: Screenshot testing for web interface
- **Load testing**: Stress testing for API endpoints
- **Security testing**: Automated security vulnerability scanning
