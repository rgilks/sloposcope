"""
AWS ECS Worker for AI Slop analysis.

Provides SQS message processing, S3 integration, and CloudWatch metrics.
"""

from . import cloudwatch_metrics, s3_client, sqs_handler

__all__ = ["sqs_handler", "s3_client", "cloudwatch_metrics"]
