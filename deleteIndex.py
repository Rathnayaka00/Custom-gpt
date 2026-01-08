import os
from dotenv import load_dotenv
import boto3

load_dotenv()

# First operation: Delete index
client = boto3.client('s3vectors', region_name=os.getenv('AWS_REGION'))
response = client.delete_index(
    vectorBucketName='ragknowledgebasev4',
    indexName='knowledgebaseindexesv4'
)
print("Delete index response:", response)

# Second operation: Delete vector bucket
# s3vectors_client = boto3.client('s3vectors', region_name=os.getenv('AWS_REGION'))
# response = s3vectors_client.delete_vector_bucket(
#     vectorBucketName='testdatabase4'
# )
# print("Delete vector bucket response:", response)