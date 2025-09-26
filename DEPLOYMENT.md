# Sloposcope Cloudflare Deployment Guide

This guide explains how to deploy your sloposcope project on Cloudflare using Workers with Python support and Next.js with OpenNext.

## Architecture Overview

- **Backend**: Python Cloudflare Worker (using Pyodide)
- **Frontend**: Next.js with OpenNext for Cloudflare Workers
- **Deployment**: Single Cloudflare Workers deployment

## Prerequisites

1. **Cloudflare Account**: Sign up at [cloudflare.com](https://cloudflare.com)
2. **Wrangler CLI**: Install Cloudflare's CLI tool
   ```bash
   npm install -g wrangler
   ```
3. **Node.js**: Version 18 or higher
4. **Python**: Version 3.11 or higher (for local development)

## Setup Instructions

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd web
npm install
```

### 2. Configure Cloudflare

```bash
# Login to Cloudflare
wrangler login

# Verify your account
wrangler whoami
```

### 3. Deploy the Python Worker

```bash
# Deploy the backend API
wrangler deploy

# Or deploy to staging
wrangler deploy --env staging
```

### 4. Deploy the Frontend

```bash
cd web

# Build the Next.js app for Cloudflare
npm run build:cf

# Deploy to Cloudflare Pages
npm run deploy
```

## Configuration

### Environment Variables

Set these in your Cloudflare dashboard or `wrangler.toml`:

```toml
[vars]
ENVIRONMENT = "production"
API_URL = "https://sloposcope.your-domain.workers.dev"
```

### Custom Domain (Optional)

1. Go to Cloudflare Dashboard → Workers & Pages
2. Select your worker
3. Go to Settings → Triggers
4. Add a custom domain

## API Endpoints

The deployed worker provides these endpoints:

- `GET /health` - Health check
- `POST /analyze` - Analyze text for AI slop
- `GET /metrics` - Get available metrics information

### Example API Usage

```bash
# Health check
curl https://sloposcope.your-domain.workers.dev/health

# Analyze text
curl -X POST https://sloposcope.your-domain.workers.dev/analyze \
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
# Start the Python worker locally
wrangler dev

# Start the Next.js frontend
cd web
npm run dev
```

### Testing

```bash
# Run Python tests
make test

# Run frontend tests
cd web
npm test
```

## Troubleshooting

### Common Issues

1. **Python Package Compatibility**: Some packages may not work with Pyodide. Check the [Pyodide compatibility list](https://pyodide.org/en/stable/usage/packages-in-pyodide.html).

2. **Memory Limits**: Cloudflare Workers have memory limits. Monitor your usage and optimize if needed.

3. **Cold Starts**: First requests may be slower due to Python initialization.

### Debugging

```bash
# View worker logs
wrangler tail

# Test locally with debugging
wrangler dev --local
```

## Performance Optimization

1. **Caching**: Use Cloudflare KV for caching analysis results
2. **CDN**: Leverage Cloudflare's global network
3. **Compression**: Enable gzip compression
4. **Minification**: Optimize JavaScript bundles

## Security Considerations

1. **Rate Limiting**: Implement rate limiting for API endpoints
2. **Input Validation**: Validate all input text
3. **CORS**: Configure CORS properly
4. **Authentication**: Add authentication if needed

## Monitoring

1. **Analytics**: Use Cloudflare Analytics
2. **Logs**: Monitor worker logs
3. **Metrics**: Track performance metrics
4. **Alerts**: Set up alerts for errors

## Scaling

Cloudflare Workers automatically scale, but consider:

1. **Resource Limits**: Monitor CPU and memory usage
2. **Concurrent Requests**: Handle high traffic
3. **Database**: Use Cloudflare D1 or external database
4. **File Storage**: Use Cloudflare R2 for large files

## Cost Optimization

1. **Free Tier**: Cloudflare Workers have generous free tier
2. **Usage Monitoring**: Track usage to avoid overages
3. **Optimization**: Optimize code for efficiency
4. **Caching**: Reduce redundant computations

## Support

- **Documentation**: [Cloudflare Workers Docs](https://developers.cloudflare.com/workers/)
- **Community**: [Cloudflare Community](https://community.cloudflare.com/)
- **Issues**: Report issues in the GitHub repository
