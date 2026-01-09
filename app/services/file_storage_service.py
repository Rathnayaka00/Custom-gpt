import mimetypes
from pathlib import Path
from typing import Optional, Tuple, Iterator

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings


_s3_client = boto3.client(
    "s3",
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
)


def _require_bucket(bucket: Optional[str]) -> str:
    if not bucket:
        raise ValueError(
            "METADATA_BUCKET_NAME is not configured. Set it in your environment/.env to enable PDF storage."
        )
    return bucket


def _ensure_bucket_exists(bucket: str) -> None:
    try:
        _s3_client.head_bucket(Bucket=bucket)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchBucket"}:
            if settings.AWS_REGION == "us-east-1":
                _s3_client.create_bucket(Bucket=bucket)
            else:
                _s3_client.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": settings.AWS_REGION},
                )
        else:
            raise


def store_uploaded_pdf(file_path: str, filename: str) -> Tuple[str, str]:
    bucket = _require_bucket(settings.METADATA_BUCKET_NAME)
    _ensure_bucket_exists(bucket)

    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"Cannot upload missing file: {file_path}")

    safe_stem = Path(filename).stem.replace("/", "_").replace("\\", "_")
    key = f"documents/{safe_stem}.pdf"

    content_type, _ = mimetypes.guess_type(filename)
    content_type = content_type or "application/pdf"

    _s3_client.upload_file(
        Filename=str(src),
        Bucket=bucket,
        Key=key,
        ExtraArgs={"ContentType": content_type},
    )
    return bucket, key


def stream_pdf_from_s3(bucket: str, key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    resp = _s3_client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"]
    for chunk in body.iter_chunks(chunk_size=chunk_size):
        if chunk:
            yield chunk


