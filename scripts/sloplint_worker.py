#!/usr/bin/env python3
"""
AI Slop Worker for AWS ECS + SQS.

Processes messages from SQS input queue, performs AI slop analysis,
and sends results to SQS output queue with CloudWatch metrics.
"""

import argparse
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sloplint.aws_worker.cloudwatch_metrics import MetricsCollector, WorkerMetrics
from sloplint.aws_worker.sqs_handler import WorkerManager


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    # Ensure log directory exists
    log_dir = "/var/log/sloplint"
    try:
        os.makedirs(log_dir, exist_ok=True)
    except PermissionError:
        # If we can't create the directory, just log to stdout
        log_dir = None

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        try:
            handlers.append(logging.FileHandler(f"{log_dir}/worker.log"))
        except (PermissionError, FileNotFoundError):
            # If we can't write to the log file, just use stdout
            pass

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def main() -> None:
    """Main worker function."""
    parser = argparse.ArgumentParser(description="AI Slop AWS Worker")
    parser.add_argument(
        "--input-queue",
        default=os.getenv("INPUT_QUEUE_URL", ""),
        help="Input SQS queue URL",
    )
    parser.add_argument(
        "--output-queue",
        default=os.getenv("OUTPUT_QUEUE_URL", ""),
        help="Output SQS queue URL",
    )
    parser.add_argument(
        "--region", default=os.getenv("AWS_REGION", "us-east-1"), help="AWS region"
    )
    parser.add_argument(
        "--log-level", default=os.getenv("LOG_LEVEL", "INFO"), help="Logging level"
    )
    parser.add_argument(
        "--max-messages", type=int, default=10, help="Max messages per poll"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=30, help="Poll interval in seconds"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Batch size for processing"
    )

    args = parser.parse_args()

    # Validate required arguments
    if not args.input_queue:
        parser.error(
            "Input queue URL is required. Provide --input-queue or set INPUT_QUEUE_URL environment variable."
        )
    if not args.output_queue:
        parser.error(
            "Output queue URL is required. Provide --output-queue or set OUTPUT_QUEUE_URL environment variable."
        )

    # Set up logging
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)

    # Debug: Print parsed arguments
    logger.info("Parsed arguments:")
    logger.info(f"  input_queue: '{args.input_queue}'")
    logger.info(f"  output_queue: '{args.output_queue}'")
    logger.info(f"  region: '{args.region}'")
    logger.info(f"  log_level: '{args.log_level}'")

    # Debug: Print all environment variables
    logger.info("Environment variables:")
    for key, value in os.environ.items():
        if "QUEUE" in key or "AWS" in key or "LOG" in key:
            logger.info(f"  {key} = {value}")

    logger.info("Starting AI Slop Worker...")
    logger.info(f"Input queue: {args.input_queue}")
    logger.info(f"Output queue: {args.output_queue}")
    logger.info(f"Region: {args.region}")

    try:
        # Configure boto3 to use LocalStack endpoints

        # Set LocalStack endpoints for all AWS services
        # localstack_config = Config(
        #     region_name=args.region,
        #     signature_version="v4",
        #     retries={"max_attempts": 3, "mode": "standard"},
        # )

        # Monkey patch boto3 session to use LocalStack
        if os.getenv("AWS_ACCESS_KEY_ID") == "test":
            # Use LocalStack endpoints - detect if running in Docker
            if os.path.exists("/.dockerenv"):
                endpoint = "http://localstack:4566"
            else:
                endpoint = "http://localhost:4566"

            os.environ["AWS_ENDPOINT_URL_SQS"] = endpoint
            os.environ["AWS_ENDPOINT_URL_S3"] = endpoint
            os.environ["AWS_ENDPOINT_URL_CLOUDWATCH"] = endpoint

        # Initialize components
        worker_manager = WorkerManager(args.input_queue, args.output_queue, args.region)
        metrics_collector = MetricsCollector(region_name=args.region)
        WorkerMetrics(metrics_collector)  # Initialize metrics (currently unused)

        logger.info("Worker components initialized successfully")

        # Run worker
        worker_manager.run_worker(
            max_messages=args.max_messages, poll_interval=args.poll_interval
        )

    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
