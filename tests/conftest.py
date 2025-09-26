"""
Pytest configuration for LocalStack testing.
"""

import os
import time

import pytest


def check_localstack_running():
    """Check if LocalStack is running."""
    try:
        import boto3

        sqs = boto3.client(
            "sqs", endpoint_url="http://localhost:4566", region_name="us-east-1"
        )
        sqs.list_queues()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def setup_localstack():
    """Set up LocalStack for testing."""
    if not check_localstack_running():
        pytest.skip(
            "LocalStack is not running. Please start it with: docker run -d --name localstack -p 4566:4566 -e SERVICES=sqs,s3,cloudwatch localstack/localstack:latest"
        )

    # Set up test environment variables
    os.environ.update(
        {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": "us-east-1",
        }
    )


@pytest.fixture(scope="session")
def localstack_health_check():
    """Ensure LocalStack is healthy before running tests."""
    max_retries = 30
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            import boto3

            sqs = boto3.client(
                "sqs", endpoint_url="http://localhost:4566", region_name="us-east-1"
            )
            sqs.list_queues()
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                pytest.fail(
                    f"LocalStack health check failed after {max_retries} attempts: {e}"
                )

    return True
