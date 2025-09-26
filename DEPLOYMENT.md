# Sloposcope Deployment Guide

This guide explains how to deploy your sloposcope project on Fly.io and Docker.

## Architecture Overview

Sloposcope supports two deployment architectures:

### Option 1: Fly.io (Recommended)

- **Backend**: FastAPI Python application
- **Frontend**: Embedded HTML/CSS/JavaScript
- **Deployment**: Single Fly.io app with Docker

### Option 2: Docker

- **Backend**: FastAPI Python application
- **Frontend**: Embedded HTML/CSS/JavaScript
- **Deployment**: Docker container with docker-compose

## Prerequisites

### For Fly.io Deployment (Recommended)

1. **Fly.io Account**: Sign up at [fly.io](https://fly.io)
2. **flyctl CLI**: Install Fly.io's CLI tool
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```
3. **Python**: Version 3.11 or higher

### For Docker Deployment

1. **Docker**: Install Docker and Docker Compose
2. **Python**: Version 3.11 or higher

## Deployment Instructions

### Option 1: Fly.io Deployment (Recommended)

#### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

#### 2. Configure Fly.io

```bash
# Login to Fly.io
flyctl auth login

# Verify your account
flyctl auth whoami
```

#### 3. Deploy to Fly.io

```bash
# Deploy using the provided script
./deploy-fly.sh

# Or deploy manually
flyctl deploy
```

Your application will be available at `https://sloposcope.fly.dev`

### Option 2: Docker Deployment

#### 1. Build and Run with Docker Compose

```bash
# Build and start services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

#### 2. Access the Application

The application will be available at `http://localhost:8000`

## Configuration

### Environment Variables

Set these in your deployment platform:

```bash
PORT=8000
ENVIRONMENT=production
```

### Custom Domain (Fly.io)

1. Go to Fly.io Dashboard → Your App
2. Go to Settings → Domains
3. Add a custom domain

## API Endpoints

The deployed application provides these endpoints:

- `GET /health` - Health check
- `POST /analyze` - Analyze text for AI slop
- `GET /metrics` - Get available metrics information

### Example API Usage

```bash
# Health check
curl https://sloposcope.fly.dev/health

# Analyze text
curl -X POST https://sloposcope.fly.dev/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here",
    "domain": "general",
    "language": "en",
    "explain": true,
    "spans": true
  }'
```

## Development Workflow

### Local Development

```bash
# Start the development server
make run

# Or with Docker
make docker-run
```

### Testing

```bash
# Run Python tests
make test

# Run unit tests only
make test-unit
```

## Troubleshooting

### Common Issues

1. **Memory Limits**: Monitor memory usage and optimize if needed
2. **Cold Starts**: First requests may be slower due to model loading
3. **Dependencies**: Ensure all Python packages are compatible

### Debugging

```bash
# View application logs (Fly.io)
flyctl logs

# View Docker logs
docker-compose logs -f
```

## Performance Optimization

1. **Caching**: Implement caching for analysis results
2. **CDN**: Leverage Fly.io's global network
3. **Compression**: Enable gzip compression
4. **Minification**: Optimize static assets

## Security Considerations

1. **Rate Limiting**: Implement rate limiting for API endpoints
2. **Input Validation**: Validate all input text
3. **CORS**: Configure CORS properly
4. **Authentication**: Add authentication if needed

## Monitoring

1. **Logs**: Monitor application logs
2. **Metrics**: Track performance metrics
3. **Alerts**: Set up alerts for errors
4. **Health Checks**: Use built-in health check endpoint

## Scaling

Both Fly.io and Docker support automatic scaling:

1. **Resource Limits**: Monitor CPU and memory usage
2. **Concurrent Requests**: Handle high traffic
3. **Database**: Use external database if needed
4. **File Storage**: Use external storage for large files

## Cost Optimization

1. **Free Tier**: Fly.io has generous free tier
2. **Usage Monitoring**: Track usage to avoid overages
3. **Optimization**: Optimize code for efficiency
4. **Caching**: Reduce redundant computations

## Support

- **Documentation**: [Fly.io Docs](https://fly.io/docs/)
- **Community**: [Fly.io Community](https://community.fly.io/)
- **Issues**: Report issues in the GitHub repository
