"""
S3 PDF Downloader Module.

Provides functionality to download PDF files from an S3 bucket.
"""
import os
import sys
import argparse
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import config

def download_pdfs_from_s3(bucket: str, prefix: str, key: str, secret: str, local_dir: str) -> int:
    """
    Downloads .pdf files from an S3 prefix to a local directory.

    Args:
        bucket: The S3 bucket name.
        prefix: The S3 prefix (folder) to download from.
        key: The AWS Access Key ID.
        secret: The AWS Secret Access Key.
        local_dir: The local directory to save files to.

    Returns:
        The count of files successfully downloaded or found locally.
    """
    print(f"Starting S3 download: s3://{bucket}/{prefix} -> {local_dir}")
    
    if not key or not secret:
        print("Error: S3 credentials (S3_AWS_ACCESS_KEY_ID, S3_AWS_SECRET_ACCESS_KEY) not provided.", file=sys.stderr)
        return 0
        
    try:
        session = boto3.Session(
            aws_access_key_id=key,
            aws_secret_access_key=secret,
        )
        s3 = session.resource("s3")
        bucket_obj = s3.Bucket(bucket)
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"S3 credentials error: {e}. Check environment variables.", file=sys.stderr)
        return 0

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    total_found = 0
    try:
        for obj in bucket_obj.objects.filter(Prefix=prefix):
            total_found += 1
            if obj.key.lower().endswith(".pdf") and obj.key != prefix:
                file_name = os.path.basename(obj.key)
                download_target = local_path / file_name
                
                if download_target.exists():
                    print(f"Skipping {obj.key}, already exists.")
                    count += 1
                    continue
                    
                try:
                    print(f"Downloading {obj.key} -> {download_target}...")
                    bucket_obj.download_file(obj.key, str(download_target))
                    count += 1
                except Exception as e:
                    print(f"Error downloading {obj.key}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error listing S3 objects. Are credentials correct? Error: {e}", file=sys.stderr)
        return 0
            
    if total_found == 0:
        print("Warning: No objects found in S3 at that bucket/prefix.")
        
    print(f"\nS3 download complete. {count} PDF files ready in {local_dir}.")
    return count

if __name__ == "__main__":
    """
    Allows standalone execution of the download script.
    """
    parser = argparse.ArgumentParser(description="Download policy PDFs from S3.")
    parser.add_argument(
        "--output-dir",
        default=config.PDF_DIR,
        help=f"Local directory to save PDFs (default: {config.PDF_DIR})"
    )
    args = parser.parse_args()

    num_downloaded = download_pdfs_from_s3(
        bucket=config.S3_BUCKET,
        prefix=config.S3_PREFIX,
        key=config.S3_AWS_ACCESS_KEY_ID,
        secret=config.S3_AWS_SECRET_ACCESS_KEY,
        local_dir=args.output_dir
    )
    
    if num_downloaded == 0:
        print("No files were downloaded.")