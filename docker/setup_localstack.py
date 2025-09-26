#!/usr/bin/env python3
"""
Setup script for LocalStack AWS resources.

Creates SQS queues and S3 bucket for testing the AI Slop worker locally.
"""

import json

import boto3
from botocore.exceptions import ClientError


def setup_aws_resources():
    """Set up AWS resources for local testing."""

    # Connect to LocalStack
    endpoint_url = "http://localhost:4566"

    # SQS client
    sqs = boto3.client("sqs", endpoint_url=endpoint_url, region_name="us-east-1")

    # S3 client
    s3 = boto3.client("s3", endpoint_url=endpoint_url, region_name="us-east-1")

    # CloudWatch client
    cloudwatch = boto3.client(
        "cloudwatch", endpoint_url=endpoint_url, region_name="us-east-1"
    )

    print("🔧 Setting up AWS resources for AI Slop Worker testing...")

    # Create SQS queues
    try:
        # Input queue for processing requests
        input_queue_response = sqs.create_queue(
            QueueName="sloplint-input-queue",
            Attributes={
                "VisibilityTimeout": "300",  # 5 minutes
                "MessageRetentionPeriod": "86400",  # 24 hours
            },
        )
        input_queue_url = input_queue_response["QueueUrl"]
        print(f"✅ Created input queue: {input_queue_url}")

        # Output queue for results
        output_queue_response = sqs.create_queue(
            QueueName="sloplint-output-queue",
            Attributes={
                "VisibilityTimeout": "300",  # 5 minutes
                "MessageRetentionPeriod": "86400",  # 24 hours
            },
        )
        output_queue_url = output_queue_response["QueueUrl"]
        print(f"✅ Created output queue: {output_queue_url}")

    except ClientError as e:
        print(f"❌ Error creating SQS queues: {e}")
        return None, None

    # Create S3 bucket
    try:
        bucket_name = "sloplint-test-bucket"
        s3.create_bucket(Bucket=bucket_name)
        print(f"✅ Created S3 bucket: {bucket_name}")

        # Create a sample text file for testing
        sample_text = """This is a sample document for testing the AI Slop worker.

It contains some repetitive content that should trigger the repetition detection.
This content is repetitive and demonstrates how the system identifies AI-generated patterns.

The worker should process this text and return detailed analysis results including:
- Density metrics
- Repetition analysis
- Coherence scores
- Template detection
- All other AI slop indicators

This is repetitive content to test the system thoroughly."""

        s3.put_object(
            Bucket=bucket_name, Key="sample_text.txt", Body=sample_text.encode("utf-8")
        )
        print("✅ Created sample text file in S3")

    except ClientError as e:
        print(f"❌ Error creating S3 bucket: {e}")
        return input_queue_url, output_queue_url

    # Create CloudWatch log group
    try:
        cloudwatch.create_log_group(logGroupName="/ecs/sloplint-worker")
        print("✅ Created CloudWatch log group: /ecs/sloplint-worker")
    except AttributeError:
        print("ℹ️  CloudWatch log groups not available in LocalStack")
    except ClientError as e:
        if "ResourceAlreadyExistsException" not in str(e):
            print(f"❌ Error creating log group: {e}")

    return input_queue_url, output_queue_url


def create_test_message(input_queue_url: str, s3_bucket: str):
    """Create a test message in the input queue."""

    sqs = boto3.client(
        "sqs", endpoint_url="http://localhost:4566", region_name="us-east-1"
    )

    # Test message using S3 reference
    message = {
        "doc_id": "test-doc-001",
        "domain": "general",
        "s3_uri": f"s3://{s3_bucket}/sample_text.txt",
        "options": {"detailed_analysis": True},
    }

    try:
        sqs.send_message(QueueUrl=input_queue_url, MessageBody=json.dumps(message))
        print("✅ Sent test message to input queue")
    except ClientError as e:
        print(f"❌ Error sending test message: {e}")


def main():
    """Main setup function."""
    print("🚀 Setting up LocalStack for AI Slop Worker testing...")

    # Set up AWS resources
    input_queue_url, output_queue_url = setup_aws_resources()

    if not input_queue_url or not output_queue_url:
        print("❌ Failed to set up AWS resources")
        return

    print("\n📋 Setup Complete!")
    print("\n📋 Queue URLs:")
    print(f"   Input Queue:  {input_queue_url}")
    print(f"   Output Queue: {output_queue_url}")

    # Create test message
    create_test_message(input_queue_url, "sloplint-test-bucket")

    print("\n🎯 Ready to test!")
    print("\n📝 Next steps:")
    print("   1. Build and run the worker: docker-compose up sloplint-worker")
    print("   2. Monitor logs: docker-compose logs -f sloplint-worker")
    print("   3. Check results: AWS CLI with LocalStack endpoints")
    print("\n💡 Environment variables for worker:")
    print("   INPUT_QUEUE_URL=http://localhost:4566/000000000000/sloplint-input-queue")
    print(
        "   OUTPUT_QUEUE_URL=http://localhost:4566/000000000000/sloplint-output-queue"
    )
    print("   AWS_REGION=us-east-1")
    print("   LOG_LEVEL=INFO")


if __name__ == "__main__":
    main()
