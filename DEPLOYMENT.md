# Sloposcope Deployment Guide

## Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/rgilks/sloposcope.git
cd sloposcope
uv sync --dev

# Download models
python -m spacy download en_core_web_sm

# Run locally
uvicorn app:app --reload
# Visit http://localhost:8000
```

### Docker Deployment

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8000
```

### Fly.io Deployment (Production)

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login to Fly.io
flyctl auth login

# Deploy
./deploy-fly.sh

# Your app will be available at https://sloposcope.fly.dev
```

## Architecture

- **Backend**: FastAPI Python application
- **Frontend**: Embedded HTML/CSS/JavaScript with Tailwind CSS
- **NLP**: spaCy with transformer models + sentence-transformers
- **Deployment**: Docker container with Fly.io support

## Configuration

### Environment Variables

- `TOKENIZERS_PARALLELISM=false` - Prevents tokenizer warnings
- `SPACY_MODEL=en_core_web_sm` - spaCy model to use
- `SENTENCE_TRANSFORMERS_MODEL=all-MiniLM-L6-v2` - Embedding model

### Domains

- `general` - General purpose text analysis
- `news` - News articles and journalism
- `qa` - Question and answer content

## Production Considerations

### Performance

- Memory usage: ~400MB peak
- Processing time: <1s for 1000 words
- Concurrent requests: Handles multiple users

### Scaling

- Fly.io: Automatic scaling based on traffic
- Docker: Manual scaling with load balancer
- Database: No database required (stateless)

### Monitoring

- Health check endpoint: `/health`
- Metrics endpoint: `/metrics`
- Logs: Available through deployment platform

## Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure spaCy model is installed
2. **Memory issues**: Increase container memory limits
3. **Performance**: Check tokenizer parallelism setting
4. **API errors**: Verify request format and content type

### Health Checks

```bash
# Check if service is running
curl http://localhost:8000/health

# Get metrics
curl http://localhost:8000/metrics
```

## Security

- CORS enabled for web interface
- No authentication required (public API)
- All processing happens locally
- No external API calls

## Maintenance

### Updates

- Pull latest changes: `git pull`
- Rebuild container: `docker-compose up --build`
- Redeploy to Fly.io: `./deploy-fly.sh`

### Monitoring

- Check logs regularly
- Monitor memory usage
- Verify health endpoints
- Test analysis accuracy

## Support

- GitHub Issues: Report bugs and feature requests
- Documentation: See README.md and DOCS.md
- API Docs: Available at `/docs` when running
