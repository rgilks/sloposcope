#!/usr/bin/env python3
"""
Run tests with LocalStack integration.
"""

import os
import subprocess
import sys
import time

import boto3


def check_localstack_running():
    """Check if LocalStack is running."""
    try:
        sqs = boto3.client(
            "sqs", endpoint_url="http://localhost:4566", region_name="us-east-1"
        )
        sqs.list_queues()
        return True
    except Exception:
        return False


def start_localstack_if_needed():
    """Start LocalStack if it's not running."""
    if check_localstack_running():
        print("✅ LocalStack is already running")
        return True

    print("🚀 Starting LocalStack for tests...")

    # Start LocalStack container
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        "localstack-test",
        "-p",
        "4566:4566",
        "-e",
        "SERVICES=sqs,s3,cloudwatch",
        "-e",
        "DEFAULT_REGION=us-east-1",
        "-e",
        "AWS_ACCESS_KEY_ID=test",
        "-e",
        "AWS_SECRET_ACCESS_KEY=test",
        "localstack/localstack:latest",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        container_id = result.stdout.strip()
        print(f"✅ Started LocalStack container: {container_id}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start LocalStack: {e}")
        return False

    # Wait for LocalStack to be ready
    print("⏳ Waiting for LocalStack to be ready...")
    max_retries = 30
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            sqs = boto3.client(
                "sqs", endpoint_url="http://localhost:4566", region_name="us-east-1"
            )
            sqs.list_queues()
            print("✅ LocalStack is ready!")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                print(
                    f"⏳ Attempt {attempt + 1}/{max_retries}: Waiting for LocalStack..."
                )
            else:
                print(f"❌ LocalStack failed to start after {max_retries} attempts: {e}")
                return False

    return False


def cleanup_localstack():
    """Clean up LocalStack container."""
    print("🧹 Cleaning up LocalStack...")

    try:
        # Stop and remove container
        subprocess.run(
            ["docker", "stop", "localstack-test"], check=True, capture_output=True
        )
        subprocess.run(
            ["docker", "rm", "localstack-test"], check=True, capture_output=True
        )
        print("✅ LocalStack cleaned up")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to clean up LocalStack: {e}")


def run_tests(test_args=None):
    """Run pytest with LocalStack tests."""
    if not start_localstack_if_needed():
        print("❌ Failed to start LocalStack. Exiting.")
        return False

    # Set up environment variables
    os.environ.update(
        {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": "us-east-1",
        }
    )

    # Build pytest command
    cmd = ["uv", "run", "python", "-m", "pytest"]

    # Add test configuration
    cmd.extend(["-c", "pytest-localstack.ini"])

    # Add test arguments if provided
    if test_args:
        cmd.extend(test_args)
    else:
        # Default: run all tests
        cmd.extend(["tests/"])

    print(f"🧪 Running tests: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return False
    finally:
        # Clean up LocalStack
        cleanup_localstack()


def main():
    """Main function."""
    # Parse command line arguments
    test_args = sys.argv[1:] if len(sys.argv) > 1 else None

    # Run tests
    success = run_tests(test_args)

    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
