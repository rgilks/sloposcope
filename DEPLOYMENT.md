# Sloposcope Deployment Guide

This comprehensive guide explains how to deploy Sloposcope across multiple platforms, from local development to production cloud environments.

## 🏗️ Architecture Overview

Sloposcope supports multiple deployment architectures:

### Option 1: Fly.io (Recommended for Production)

- **Backend**: FastAPI Python application with transformer models
- **Frontend**: Embedded HTML/CSS/JavaScript with Tailwind CSS
- **Deployment**: Single Fly.io app with Docker
- **Benefits**: Global CDN, automatic scaling, easy SSL, generous free tier

### Option 2: Docker (Local Development & Self-Hosted)

- **Backend**: FastAPI Python application
- **Frontend**: Embedded HTML/CSS/JavaScript with Tailwind CSS
- **Deployment**: Docker container with docker-compose
- **Benefits**: Consistent environment, easy local development


## 📋 Prerequisites

### For Fly.io Deployment (Recommended)

1. **Fly.io Account**: Sign up at [fly.io](https://fly.io)
2. **flyctl CLI**: Install Fly.io's CLI tool
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```
3. **Python**: Version 3.11 or higher
4. **Docker**: For building the container image

### For Docker Deployment

1. **Docker**: Install Docker and Docker Compose
2. **Python**: Version 3.11 or higher
3. **Git**: For cloning the repository


## 🚀 Deployment Instructions

### Option 1: Fly.io Deployment (Recommended)

#### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-org/sloposcope.git
cd sloposcope

# Install Python dependencies
uv sync --dev

# Download required models
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm
```

#### 2. Configure Fly.io

```bash
# Login to Fly.io
flyctl auth login

# Verify your account
flyctl auth whoami

# Initialize Fly.io app (if not already done)
flyctl launch --no-deploy
```

#### 3. Deploy to Fly.io

```bash
# Deploy using the provided script
./deploy-fly.sh

# Or deploy manually
flyctl deploy

# Check deployment status
flyctl status
```

Your application will be available at `https://sloposcope.fly.dev`

#### 4. Custom Domain Setup

```bash
# Add custom domain
flyctl certs add your-domain.com

# Check certificate status
flyctl certs show your-domain.com
```

### Option 2: Docker Deployment

#### 1. Build and Run with Docker Compose

```bash
# Build and start services
docker-compose up --build

# Or run in background
docker-compose up -d --build

# Check logs
docker-compose logs -f
```

#### 2. Access the Application

The application will be available at `http://localhost:8000`

#### 3. Production Docker Setup

```bash
# Build production image
docker build -f Dockerfile -t sloposcope:latest .

# Run production container
docker run -d \
  --name sloposcope \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  sloposcope:latest
```


## ⚙️ Configuration

### Environment Variables

Set these in your deployment platform:

```bash
# Application Configuration
PORT=8000
ENVIRONMENT=production
LOG_LEVEL=INFO

# Model Configuration
SPACY_MODEL=en_core_web_trf
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2

# Performance Configuration
MAX_WORKERS=4
WORKER_TIMEOUT=30

```

### Fly.io Configuration

Update `fly.toml` for your specific needs:

```toml
[app]
  app = "sloposcope"
  primary_region = "ord"

[build]

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 1024

[env]
  ENVIRONMENT = "production"
  LOG_LEVEL = "INFO"
```

### Docker Configuration

Update `docker-compose.yml` for your environment:

```yaml
version: "3.8"

services:
  sloposcope:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

## 🔧 Development Workflow

### Local Development

```bash
# Start the development server
make run

# Or with Docker
make docker-run

# Run tests
make test

# Run linting
make lint
```

### Testing Deployments

```bash
# Test Fly.io deployment
curl https://sloposcope.fly.dev/health

# Test Docker deployment
curl http://localhost:8000/health

```

## 🔍 Monitoring and Observability

### Health Checks

All deployments include health check endpoints:

```bash
# Basic health check
curl https://sloposcope.fly.dev/health

# Detailed metrics
curl https://sloposcope.fly.dev/metrics
```

### Logging

#### Fly.io Logs

```bash
# View application logs
flyctl logs

# Follow logs in real-time
flyctl logs -f

# View logs for specific app
flyctl logs -a sloposcope
```

#### Docker Logs

```bash
# View container logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f sloposcope
```


### Metrics

#### Fly.io Metrics

```bash
# View app metrics
flyctl metrics

# View specific metrics
flyctl metrics --type cpu,memory
```


## 🚨 Troubleshooting

### Common Issues

#### 1. Memory Issues

**Symptoms**: Application crashes, out of memory errors

**Solutions**:

```bash
# Increase memory allocation (Fly.io)
flyctl scale memory 2048

# Increase Docker memory limit
docker run -m 2g sloposcope:latest

# Monitor memory usage
flyctl metrics --type memory
```

#### 2. Model Loading Issues

**Symptoms**: Slow startup, model not found errors

**Solutions**:

```bash
# Check if models are installed
python -c "import spacy; spacy.load('en_core_web_trf')"

# Download missing models
python -m spacy download en_core_web_trf

# Pre-build models in Docker
docker build --build-arg DOWNLOAD_MODELS=true -t sloposcope:latest .
```

#### 3. Performance Issues

**Symptoms**: Slow response times, high CPU usage

**Solutions**:

```bash
# Scale horizontally (Fly.io)
flyctl scale count 2

# Optimize Docker resources
docker run --cpus=2 --memory=2g sloposcope:latest

# Monitor performance
flyctl metrics --type cpu,memory,latency
```


### Debugging Commands

```bash
# Debug Fly.io deployment
flyctl ssh console

# Debug Docker container
docker exec -it sloposcope-container /bin/bash

```

## 🔒 Security Considerations

### 1. Input Validation

- Validate all input text for length and content
- Implement rate limiting for API endpoints
- Sanitize user inputs to prevent injection attacks

### 2. Authentication & Authorization

```bash
# Add API key authentication (optional)
export API_KEY=your-secret-key

# Configure CORS properly
export CORS_ORIGINS=https://yourdomain.com
```

### 3. Data Protection

- No sensitive data persisted unless explicitly configured
- Use HTTPS for all communications
- Implement proper logging without exposing sensitive information


## 📈 Performance Optimization

### 1. Caching

```bash
# Enable Redis caching (optional)
export REDIS_URL=redis://localhost:6379

# Enable model caching
export CACHE_MODELS=true
```

### 2. CDN

Fly.io automatically provides global CDN. For other deployments:

```bash
# Configure CloudFlare or similar CDN
# Set appropriate cache headers
# Enable compression
```

### 3. Resource Optimization

```bash
# Optimize Docker image size
docker build --target production -t sloposcope:latest .

# Use multi-stage builds
# Minimize dependencies
# Use Alpine Linux base images
```

## 💰 Cost Optimization

### Fly.io

- Use shared CPU instances for development
- Enable auto-stop/start for non-production apps
- Monitor usage with `flyctl metrics`


### Docker

- Use multi-stage builds to reduce image size
- Implement health checks to restart unhealthy containers
- Use resource limits to prevent runaway processes

## 🔄 CI/CD Pipeline

### GitHub Actions Example

```yaml
name: Deploy to Fly.io

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: superfly/flyctl-actions/setup-flyctl@master
      - run: flyctl deploy --remote-only
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```

### Docker Hub Example

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and push
        uses: docker/build-push-action@v3
        with:
          context: .
          push: true
          tags: your-org/sloposcope:latest
```

## 📚 Additional Resources

- [Fly.io Documentation](https://fly.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [spaCy Model Installation](https://spacy.io/usage/models)

## 🆘 Support

For deployment issues:

1. Check the troubleshooting section above
2. Review application logs
3. Check platform-specific documentation
4. Open an issue in the GitHub repository
5. Join the community discussions
