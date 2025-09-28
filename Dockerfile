# Use Python 3.11 alpine image for minimal size
FROM python:3.11-alpine

# Set environment variables for CPU-only operation
ENV PYTORCH_NO_CUDA=1
ENV TOKENIZERS_PARALLELISM=false
ENV FORCE_CUDA=0

# Set working directory
WORKDIR /app

# Install system dependencies (including build tools and Linux headers)
RUN apk add --no-cache \
    curl \
    build-base \
    linux-headers \
    && rm -rf /var/cache/apk/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies (production only)
# Use a virtual environment to avoid root installation
RUN uv venv && uv sync --frozen --no-dev

# Copy application code (excluding startup script)
COPY sloplint/ ./sloplint/
COPY app.py ./
COPY templates/ ./templates/

# Copy startup script to root directory AFTER copying app code
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

# Create non-root user for security
RUN adduser -D -s /bin/sh -u 1000 app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check with timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application directly
ENTRYPOINT ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
