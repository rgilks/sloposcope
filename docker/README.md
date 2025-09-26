# AI Slop Worker - AWS Deployment

This directory contains the Docker and infrastructure configuration for deploying the Sloposcope AI Text Analysis tool as an AWS ECS worker with SQS integration.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Queue   │───▶│   ECS Worker    │───▶│  Output Queue   │
│     (SQS)       │    │   (Fargate)     │    │     (SQS)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  CloudWatch     │
                       │   Metrics       │
                       └─────────────────┘
```

## 🧩 Components

### 1. AWS Worker (`sloplint/aws_worker/`)

- **SQS Handler**: Message polling, processing, and queue management
- **S3 Client**: Text storage and retrieval from S3
- **CloudWatch Metrics**: Metrics collection and publishing

### 2. Worker Script (`scripts/sloplint_worker.py`)

- Main worker entry point
- Configuration via environment variables
- Health checks and error handling

### 3. Docker Configuration

- **Dockerfile**: Multi-stage build for production
- **docker-compose.yml**: Local development setup
- **terraform.tf**: Infrastructure as Code

## ⚙️ Environment Variables

```bash
# AWS Configuration
AWS_REGION=us-east-1
INPUT_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789/input-queue
OUTPUT_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789/output-queue

# Worker Configuration
LOG_LEVEL=INFO
MAX_MESSAGES=10
POLL_INTERVAL=30
BATCH_SIZE=5

# Model Configuration
SPACY_MODEL=en_core_web_trf
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
```

## 📨 Message Format

### Input Message

```json
{
  "doc_id": "unique-document-id",
  "domain": "news",
  "text": "Full text content to analyze",
  "s3_uri": "s3://bucket/path/to/file.txt",
  "prompt": "Optional analysis prompt",
  "references": ["s3://bucket/ref1.txt", "s3://bucket/ref2.txt"],
  "options": {
    "custom_thresholds": {...}
  }
}
```

### Output Message

```json
{
  "doc_id": "unique-document-id",
  "status": "ok",
  "result": {
    "version": "1.0",
    "domain": "news",
    "slop_score": 0.347,
    "confidence": 1.0,
    "level": "Watch",
    "metrics": {
      "density": {"value": 0.5, "perplexity": 25.0, ...},
      "repetition": {"value": 0.3, "compression_ratio": 0.4, ...},
      // ... all 11 metrics
    },
    "spans": [
      {
        "start": 120,
        "end": 165,
        "axis": "repetition",
        "note": "Repeated sentence stem"
      }
    ],
    "timings_ms": {"total": 500, "nlp": 200, "features": 300}
  },
  "error": null
}
```

## 🚀 Deployment Options

### 1. Docker Compose (Local Development)

```bash
# Start with LocalStack for local SQS/S3
docker-compose up -d localstack

# Build and run worker
docker-compose up sloplint-worker
```

### 2. AWS ECS (Production)

#### Prerequisites

- AWS CLI configured with appropriate credentials
- ECR repository created
- SQS queues created (input and output)
- VPC and subnets configured

#### Deploy with Terraform

```bash
# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var="input_queue_url=..." -var="output_queue_url=..."

# Deploy infrastructure
terraform apply

# Build and push Docker image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker build -f docker/Dockerfile -t sloplint-worker .
docker tag sloplint-worker:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/sloplint-worker:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/sloplint-worker:latest

# Update ECS service
aws ecs update-service --cluster sloplint-worker-cluster --service sloplint-worker-service --force-new-deployment
```

## 📊 Monitoring

### CloudWatch Metrics

The worker publishes the following metrics:

- **MessageProcessingTime**: Time to process each message (ms)
- **MessagesProcessed**: Number of successfully processed messages
- **MessagesFailed**: Number of failed messages
- **FeatureExtractionTime/{feature}**: Time for each feature extractor
- **SlopScore**: AI slop scores distribution
- **AverageSlopScore**: Batch average scores
- **QueueDepth**: SQS queue depths
- **WorkerErrors**: Error counts by type

### Logging

Logs are sent to CloudWatch Logs group `/ecs/sloplint-worker`.

### Health Checks

- ECS health checks every 30 seconds
- Graceful shutdown on SIGTERM
- Startup period of 60 seconds

## 📈 Scaling

### Auto Scaling

Configure ECS Service Auto Scaling based on:

- SQS queue depth
- CPU/Memory utilization
- Custom CloudWatch metrics

### Performance Tuning

- **CPU**: 512 vCPU (configurable in Terraform)
- **Memory**: 2048 MB (configurable in Terraform)
- **Batch Size**: 5 messages (configurable via env var)
- **Poll Interval**: 30 seconds (configurable via env var)

## 🔒 Security

### IAM Permissions

The worker uses least-privilege IAM roles:

- **Task Execution Role**: ECS task execution permissions
- **Task Role**: SQS, S3, and CloudWatch permissions

### Network Security

- Security groups allow only necessary traffic
- VPC endpoints for AWS services (optional)
- Private subnets with NAT gateway

### Data Protection

- S3 server-side encryption (SSE-S3 or SSE-KMS)
- SQS message encryption at rest
- No sensitive data in logs

## 🐛 Troubleshooting

### Common Issues

1. **High Memory Usage**

   - Reduce batch size
   - Increase memory allocation
   - Check for memory leaks in feature extractors

2. **SQS Message Processing Errors**

   - Check message format
   - Verify S3 permissions
   - Check CloudWatch logs for detailed errors

3. **Model Download Issues**

   - Ensure internet access for initial model downloads
   - Use pre-built Docker images
   - Check disk space in container

4. **Performance Issues**
   - Monitor CloudWatch metrics
   - Adjust ECS task size
   - Optimize feature extraction

### Debugging Commands

```bash
# Check ECS task status
aws ecs describe-tasks --cluster sloplint-worker-cluster --tasks $(aws ecs list-tasks --cluster sloplint-worker-cluster --query taskArns[0])

# View CloudWatch logs
aws logs tail /ecs/sloplint-worker --follow

# Check SQS queue depth
aws sqs get-queue-attributes --queue-url <queue-url> --attribute-names ApproximateNumberOfMessages

# Test worker locally
docker run -e INPUT_QUEUE_URL=... -e OUTPUT_QUEUE_URL=... sloplint-worker
```

## 💰 Cost Optimization

- **Fargate Spot**: Use spot instances for non-critical workloads
- **Auto Scaling**: Scale down during low traffic periods
- **Monitoring**: Set up billing alerts for unexpected costs
- **Data Transfer**: Use VPC endpoints to reduce data transfer costs

## 🛠️ Development

### Local Testing

```bash
# Run with local SQS
docker-compose up localstack
# Wait for LocalStack to start, then create queues
# Set environment variables and run worker
```

### CI/CD Pipeline

```yaml
# Example GitHub Actions workflow
- name: Build and Push Docker Image
  run: |
    docker build -f docker/Dockerfile -t sloplint-worker .
    docker tag sloplint-worker $ECR_REPOSITORY:latest
    docker push $ECR_REPOSITORY:latest

- name: Deploy to ECS
  run: |
    aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_SERVICE --force-new-deployment
```

## 📚 Additional Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [AWS SQS Documentation](https://docs.aws.amazon.com/sqs/)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest)

## 🆘 Support

For issues and questions:

- Check CloudWatch logs first
- Review ECS task events
- Monitor CloudWatch metrics
- Check SQS message visibility timeout settings

The worker is designed to be resilient and self-healing with proper monitoring and alerting in place.
