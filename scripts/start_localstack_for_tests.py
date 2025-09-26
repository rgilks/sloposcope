#!/usr/bin/env python3
"""
Start LocalStack for testing purposes.
"""

import subprocess
import sys
import time

import boto3


def check_localstack_running():
    """Check if LocalStack is already running."""
    try:
        sqs = boto3.client(
            "sqs", endpoint_url="http://localhost:4566", region_name="us-east-1"
        )
        sqs.list_queues()
        return True
    except Exception:
        return False


def start_localstack():
    """Start LocalStack container."""
    if check_localstack_running():
        print("✅ LocalStack is already running")
        return True

    print("🚀 Starting LocalStack for testing...")

    # Check if docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not available. Please install Docker and try again.")
        return False

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
        print(f"Error output: {e.stderr}")
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
                print(
                    f"❌ LocalStack failed to start after {max_retries} attempts: {e}"
                )
                return False

    return False


def stop_localstack():
    """Stop LocalStack container."""
    print("🛑 Stopping LocalStack...")

    try:
        # Stop and remove container
        subprocess.run(
            ["docker", "stop", "localstack-test"], check=True, capture_output=True
        )
        subprocess.run(
            ["docker", "rm", "localstack-test"], check=True, capture_output=True
        )
        print("✅ LocalStack stopped and removed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to stop LocalStack: {e}")
        return False

    return True


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python start_localstack_for_tests.py [start|stop|status]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "start":
        success = start_localstack()
        sys.exit(0 if success else 1)
    elif command == "stop":
        success = stop_localstack()
        sys.exit(0 if success else 1)
    elif command == "status":
        if check_localstack_running():
            print("✅ LocalStack is running")
            sys.exit(0)
        else:
            print("❌ LocalStack is not running")
            sys.exit(1)
    else:
        print("Invalid command. Use 'start', 'stop', or 'status'")
        sys.exit(1)


if __name__ == "__main__":
    main()
