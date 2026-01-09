import json
from datetime import datetime
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings


s3_client = boto3.client(
    's3',
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
)

METADATA_BUCKET_NAME = settings.METADATA_BUCKET_NAME
METADATA_PREFIX = "metadata/"

def _get_metadata_key(s3_key: str) -> str:
    safe_key = s3_key.replace("/", "_").replace("\\", "_")
    return f"{METADATA_PREFIX}{safe_key}.json"

def _require_bucket():
    if not METADATA_BUCKET_NAME:
        raise ValueError(
            "METADATA_BUCKET_NAME is not configured. Set it in your environment/.env to enable metadata storage."
        )

def _ensure_bucket_exists():
    _require_bucket()
    try:
        s3_client.head_bucket(Bucket=METADATA_BUCKET_NAME)
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code in {'404', 'NoSuchBucket'}:
            try:
                if settings.AWS_REGION == 'us-east-1':
                    s3_client.create_bucket(Bucket=METADATA_BUCKET_NAME)
                else:
                    s3_client.create_bucket(
                        Bucket=METADATA_BUCKET_NAME,
                        CreateBucketConfiguration={'LocationConstraint': settings.AWS_REGION}
                    )
            except Exception as create_error:
                print(f"Error creating bucket: {create_error}")
        else:
            print(f"Error checking bucket: {e}")

def store_document_metadata(
    s3_key: str,
    filename: str,
    content_type: str,
    file_size: int,
    total_chunks: int,
    document_bucket: Optional[str] = None,
    document_key: Optional[str] = None,
    upload_date: Optional[datetime] = None
) -> bool:
    try:
        _require_bucket()
        _ensure_bucket_exists()

        if upload_date is None:
            upload_date = datetime.utcnow()

        metadata = {
            's3_key': s3_key,
            'filename': filename,
            'content_type': content_type,
            'file_size': file_size,
            'total_chunks': total_chunks,
            'document_bucket': document_bucket,
            'document_key': document_key,
            'upload_date': upload_date.isoformat(),
            'status': 'active',
            'created_at': datetime.utcnow().isoformat()
        }

        metadata_key = _get_metadata_key(s3_key)

        s3_client.put_object(
            Bucket=METADATA_BUCKET_NAME,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2),
            ContentType='application/json'
        )
        return True

    except Exception as e:
        print(f"Error storing document metadata: {e}")
        raise

def get_document_metadata(s3_key: str) -> Optional[Dict]:
    try:
        _require_bucket()
        metadata_key = _get_metadata_key(s3_key)
        response = s3_client.get_object(Bucket=METADATA_BUCKET_NAME, Key=metadata_key)
        metadata_content = response['Body'].read().decode('utf-8')
        return json.loads(metadata_content)
    except ClientError as e:
        code = e.response.get('Error', {}).get('Code')
        if code in {'NoSuchKey', '404'}:
            return None
        print(f"Error retrieving document metadata: {e}")
        raise
    except Exception as e:
        print(f"Error retrieving document metadata: {e}")
        raise

def list_all_documents() -> List[Dict]:
    try:
        _require_bucket()
        _ensure_bucket_exists()
        response = s3_client.list_objects_v2(Bucket=METADATA_BUCKET_NAME, Prefix=METADATA_PREFIX)
        documents: List[Dict] = []
        if 'Contents' in response:
            for obj in response['Contents']:
                try:
                    metadata_response = s3_client.get_object(Bucket=METADATA_BUCKET_NAME, Key=obj['Key'])
                    metadata_content = metadata_response['Body'].read().decode('utf-8')
                    metadata = json.loads(metadata_content)
                    if metadata.get('status') == 'active':
                        documents.append(metadata)
                except Exception as e:
                    print(f"Error reading metadata file {obj['Key']}: {e}")
                    continue
        return documents
    except Exception as e:
        print(f"Error listing documents: {e}")
        raise

def mark_document_deleted(s3_key: str) -> bool:
    try:
        _require_bucket()
        metadata = get_document_metadata(s3_key)
        if not metadata:
            return False
        metadata['status'] = 'deleted'
        metadata['deleted_at'] = datetime.utcnow().isoformat()
        metadata_key = _get_metadata_key(s3_key)
        s3_client.put_object(
            Bucket=METADATA_BUCKET_NAME,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2),
            ContentType='application/json'
        )
        return True
    except Exception as e:
        print(f"Error marking document as deleted: {e}")
        raise

def delete_document_metadata(s3_key: str) -> bool:
    try:
        _require_bucket()
        metadata_key = _get_metadata_key(s3_key)
        s3_client.delete_object(Bucket=METADATA_BUCKET_NAME, Key=metadata_key)
        return True
    except Exception as e:
        print(f"Error deleting document metadata: {e}")
        raise

def get_vector_keys_by_s3_key(s3_key: str) -> List[str]:
    try:
        _require_bucket()
        metadata = get_document_metadata(s3_key)
        if not metadata:
            return []
        total_chunks = metadata.get('total_chunks', 0)
        vector_keys = []
        for i in range(total_chunks):
            vector_keys.append(f"{s3_key}-chunk{str(i).zfill(4)}")
        return vector_keys
    except Exception as e:
        print(f"Error generating vector keys: {e}")
        return []


