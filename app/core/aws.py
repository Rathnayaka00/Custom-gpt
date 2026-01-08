import boto3
from botocore.config import Config as BotoConfig
from app.core.config import settings

session = boto3.Session(
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION
)

s3_vectors_client = session.client("s3vectors")

bedrock_boto_config = BotoConfig(
    retries={
        "max_attempts": settings.BEDROCK_MAX_RETRIES,
        "mode": settings.BEDROCK_RETRY_MODE,
    },
    connect_timeout=settings.BEDROCK_CONNECT_TIMEOUT_SECONDS,
    read_timeout=settings.BEDROCK_READ_TIMEOUT_SECONDS,
)

bedrock_runtime_client = session.client(
    service_name=settings.BEDROCK_SERVICE_NAME,
    region_name=settings.AWS_REGION,
    config=bedrock_boto_config,
)