# Testing Guide

This document describes how to run tests for the AI Slop project, including LocalStack integration tests.

## Test Types

### Unit Tests
- **Location**: `tests/test_cli.py`
- **Purpose**: Test core functionality without external dependencies
- **Speed**: Fast (< 10 seconds)
- **Dependencies**: None

### AWS Worker Tests
- **Location**: `tests/test_aws_worker.py`
- **Purpose**: Test AWS integration using LocalStack
- **Speed**: Medium (30-60 seconds)
- **Dependencies**: LocalStack, Docker

## Quick Start

### Run All Tests
```bash
make test
```

### Run Unit Tests Only
```bash
make test-unit
```

### Run AWS Worker Tests
```bash
make test-aws
```

## Manual Testing

### Start LocalStack
```bash
python scripts/start_localstack_for_tests.py start
```

### Run Tests with LocalStack
```bash
python scripts/run_tests_with_localstack.py
```

### Stop LocalStack
```bash
python scripts/start_localstack_for_tests.py stop
```

## Pre-commit Hooks

The project includes pre-commit hooks that automatically run tests:

### Install Pre-commit Hooks
```bash
make pre-commit
```

### Run Pre-commit Hooks Manually
```bash
make pre-commit-run
```

### Hook Behavior
- **Unit tests**: Run on every commit (fast)
- **AWS Worker tests**: Run only when AWS-related files are changed
- **Linting**: Run on every commit
- **Type checking**: Run on every commit

## Test Configuration

### Pytest Configuration
- **Main config**: `pyproject.toml`
- **LocalStack config**: `pytest-localstack.ini`
- **Test discovery**: `tests/` directory

### Environment Variables
Tests automatically set these environment variables:
```bash
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
AWS_DEFAULT_REGION=us-east-1
```

## Test Structure

### Fixtures
- `localstack_endpoint`: LocalStack URL
- `aws_credentials`: Test AWS credentials
- `localstack_resources`: SQS queues and S3 buckets for testing

### Test Classes
- `TestSlopMessage`: Message data structure tests
- `TestSlopResult`: Result data structure tests
- `TestS3TextClient`: S3 integration tests
- `TestSQSPoller`: SQS message handling tests
- `TestWorkerManager`: End-to-end worker tests
- `TestMetricsCollector`: CloudWatch metrics tests
- `TestEndToEndIntegration`: Full workflow tests

## Docker Testing

### Test with Docker Compose
```bash
make docker-test
```

### Clean Up Docker
```bash
make docker-clean
```

## Continuous Integration

The CI pipeline runs:
1. Linting (`ruff`, `black`)
2. Type checking (`mypy`)
3. Unit tests
4. AWS Worker tests (if AWS files changed)

## Troubleshooting

### LocalStack Not Starting
```bash
# Check if Docker is running
docker --version

# Check if port 4566 is available
lsof -i :4566

# Clean up existing containers
make clean-localstack
```

### Tests Failing
```bash
# Check LocalStack status
python scripts/start_localstack_for_tests.py status

# Run tests with verbose output
python scripts/run_tests_with_localstack.py -v

# Run specific test
python scripts/run_tests_with_localstack.py tests/test_aws_worker.py::TestSlopMessage::test_message_creation -v
```

### Pre-commit Hooks Failing
```bash
# Update pre-commit hooks
pre-commit autoupdate

# Run specific hook
pre-commit run pytest-unit --all-files
```

## Performance

### Test Execution Times
- Unit tests: ~5-10 seconds
- AWS Worker tests: ~30-60 seconds
- Full test suite: ~60-90 seconds

### Optimization Tips
- Use `make test-unit` for quick feedback during development
- Use `make test-aws` only when working on AWS components
- Pre-commit hooks are optimized to run only relevant tests

## Coverage

Test coverage reports are generated in:
- Terminal output
- HTML: `htmlcov/index.html`
- XML: `coverage.xml`

Current coverage:
- Overall: ~21%
- AWS Worker components: ~15-25%
- Core functionality: ~20-30%
