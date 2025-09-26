"""
S3 client for AI Slop AWS worker.

Handles text storage, retrieval, and file management in S3.
"""

import gzip
import logging
import os

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logging.warning("boto3 not available. AWS functionality will be limited.")

logger = logging.getLogger(__name__)


class S3TextClient:
    """Handles text storage and retrieval from S3."""

    def __init__(self, bucket_name: str, region_name: str = "us-east-1"):
        """Initialize S3 client."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for AWS functionality")

        self.bucket_name = bucket_name
        # Use LocalStack endpoint if test credentials are detected
        if os.getenv("AWS_ACCESS_KEY_ID") == "test":
            # Detect if running in Docker container
            if os.path.exists("/.dockerenv") or os.getenv("CONTAINER") == "docker":
                endpoint_url = "http://localstack:4566"
            else:
                endpoint_url = "http://localhost:4566"
            self.s3_client = boto3.client(
                "s3", region_name=region_name, endpoint_url=endpoint_url
            )
        else:
            self.s3_client = boto3.client("s3", region_name=region_name)
        self.logger = logging.getLogger(__name__)

    def load_text(self, s3_uri: str) -> str:
        """Load text content from S3 URI."""
        try:
            # Parse S3 URI (e.g., s3://bucket/path/to/file.txt)
            if not s3_uri.startswith("s3://"):
                raise ValueError(f"Invalid S3 URI: {s3_uri}")

            s3_path = s3_uri[5:]  # Remove 's3://' prefix
            bucket, key = s3_path.split("/", 1)

            # Get object from S3
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()

            # Check if content is gzipped
            if key.endswith(".gz") or response.get("ContentEncoding") == "gzip":
                content = gzip.decompress(content)

            # Decode as UTF-8 text
            text = content.decode("utf-8")

            self.logger.info(f"Loaded text from {s3_uri} ({len(text)} characters)")
            return text

        except ClientError as e:
            self.logger.error(f"Error loading from S3 {s3_uri}: {e}")
            raise ValueError(f"Could not load from S3: {s3_uri}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error loading from S3 {s3_uri}: {e}")
            raise ValueError(f"Could not load from S3: {s3_uri}") from e

    def save_text(self, text: str, s3_uri: str, compress: bool = False) -> bool:
        """Save text content to S3 URI."""
        try:
            # Parse S3 URI
            if not s3_uri.startswith("s3://"):
                raise ValueError(f"Invalid S3 URI: {s3_uri}")

            s3_path = s3_uri[5:]  # Remove 's3://' prefix
            bucket, key = s3_path.split("/", 1)

            # Prepare content
            content = text.encode("utf-8")

            if compress:
                content = gzip.compress(content)
                if not key.endswith(".gz"):
                    key += ".gz"

            # Upload to S3
            if compress:
                self.s3_client.put_object(
                    Bucket=bucket, Key=key, Body=content, ContentEncoding="gzip"
                )
            else:
                self.s3_client.put_object(Bucket=bucket, Key=key, Body=content)

            self.logger.info(f"Saved text to {s3_uri} ({len(text)} characters)")
            return True

        except ClientError as e:
            self.logger.error(f"Error saving to S3 {s3_uri}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error saving to S3 {s3_uri}: {e}")
            return False

    def save_json_result(self, result: dict, s3_uri: str) -> bool:
        """Save JSON result to S3."""
        import json

        try:
            # Convert result to JSON string
            json_content = json.dumps(result, indent=2, ensure_ascii=False)

            # Save as text
            return self.save_text(json_content, s3_uri)

        except Exception as e:
            self.logger.error(f"Error saving JSON to S3 {s3_uri}: {e}")
            return False

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> list[str]:
        """List objects in S3 bucket with optional prefix."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=prefix, MaxKeys=max_keys
            )

            objects = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    key = obj["Key"]
                    objects.append(f"s3://{self.bucket_name}/{key}")

            self.logger.info(f"Found {len(objects)} objects with prefix '{prefix}'")
            return objects

        except ClientError as e:
            self.logger.error(f"Error listing S3 objects: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error listing S3 objects: {e}")
            return []

    def object_exists(self, s3_uri: str) -> bool:
        """Check if an S3 object exists."""
        try:
            # Parse S3 URI
            if not s3_uri.startswith("s3://"):
                return False

            s3_path = s3_uri[5:]  # Remove 's3://' prefix
            bucket, key = s3_path.split("/", 1)

            # Check if object exists
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            self.logger.error(f"Error checking S3 object {s3_uri}: {e}")
            return False
        except Exception:
            return False
