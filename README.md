# AI Slop CLI

A comprehensive command-line tool for detecting AI-generated text patterns and measuring "slop" across multiple dimensions.

## Features

- **Multi-dimensional Analysis**: 11 different metrics including density, repetition, coherence, templatedness, and more
- **Configurable Scoring**: Domain-specific weighting and calibration
- **AWS Integration**: Full ECS + SQS worker deployment
- **High Performance**: Optimized for batch processing and scaling
- **Production Ready**: Comprehensive error handling and monitoring

## Quick Start

```bash
# Install locally
pip install -e .

# Analyze text
sloplint analyze "Your text here" --domain general

# AWS deployment
cd docker && docker-compose up
```

## Testing

The project includes comprehensive testing with LocalStack integration:

```bash
# Run all tests
make test

# Run unit tests only (fast)
make test-unit

# Run AWS Worker tests with LocalStack
make test-aws

# Set up development environment
make dev-setup
```

See [TESTING.md](TESTING.md) for detailed testing instructions.

## Documentation

See the full documentation in the `docs/` directory for detailed usage instructions, API reference, and deployment guides.
