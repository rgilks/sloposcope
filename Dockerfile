# Use Python 3.11 slim image with minimal dependencies
FROM python:3.11-slim

# Set environment variables for CPU-only operation
ENV PYTORCH_NO_CUDA=1
ENV TOKENIZERS_PARALLELISM=false
ENV FORCE_CUDA=0

# Set working directory
WORKDIR /app

# Install minimal system dependencies (only what's needed for runtime)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies (production only, excluding all dev dependencies)
# This ensures only runtime dependencies are included in the Docker image
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Download spaCy model using pip (more reliable method)
RUN uv run pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check with timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
