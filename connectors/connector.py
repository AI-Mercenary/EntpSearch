import os
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

BUCKET_NAME = "bucketdatafyndo.ai"  # replace with your bucket name

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

def list_all_files():
    """List all files in the S3 bucket with pagination."""
    continuation_token = None
    files = []

    while True:
        if continuation_token:
            response = s3.list_objects_v2(Bucket=BUCKET_NAME, ContinuationToken=continuation_token)
        else:
            response = s3.list_objects_v2(Bucket=BUCKET_NAME)

        contents = response.get("Contents", [])
        for obj in contents:
            files.append({
                "Key": obj["Key"],
                "LastModified": obj["LastModified"]
            })

        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break

    return files

def get_file_content(key):
    """Get content of the file as text, return None if binary/unreadable."""
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    data = obj['Body'].read()
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return None

if __name__ == "__main__":
    files = list_all_files()
    print(f"Found {len(files)} files in bucket '{BUCKET_NAME}'")
    for f in files[:5]:  # print first 5 for sanity check
        print(f"File: {f['Key']} | LastModified: {f['LastModified']}")
