# Multi-stage Dockerfile for AI Slop Worker
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -e .

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download sentence transformers model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Production stage
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user and log directory
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /var/log/sloplint && \
    chown -R app:app /var/log/sloplint
USER app
WORKDIR /home/app

# Copy installed packages from base stage
COPY --from=base /home/app/.local /home/app/.local

# Add local bin to PATH
ENV PATH="/home/app/.local/bin:${PATH}"

# Copy worker script
COPY --chown=app:app scripts/sloplint_worker.py /home/app/worker.py

# Set environment variables
ENV PYTHONPATH="/home/app"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run worker
CMD ["python", "/home/app/worker.py", "--input-queue", "http://localstack:4566/000000000000/sloplint-input-queue", "--output-queue", "http://localstack:4566/000000000000/sloplint-output-queue"]
