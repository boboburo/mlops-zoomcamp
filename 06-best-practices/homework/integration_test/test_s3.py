import os

import boto3

s3_endpoint = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
s3_client = boto3.client('s3', )