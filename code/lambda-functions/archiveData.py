import boto3
import logging
import urllib.parse

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda function to move files from one S3 prefix to another.
    Expects the following parameters in the event:
    - bucket: The S3 bucket name
    - source_prefix: The source prefix (e.g., 'landing-data/')
    - destination_prefix: The destination prefix (e.g., 'archive/invalid-data/')
    """
    try:
        # Extract parameters from the event
        bucket = event.get('bucket')
        source_prefix = event.get('source_prefix')
        destination_prefix = event.get('destination_prefix')

        if not bucket or not source_prefix or not destination_prefix:
            raise ValueError("Missing required parameters: bucket, source_prefix, destination_prefix")

        logger.info(f"Moving files from {source_prefix} to {destination_prefix} in bucket {bucket}")

        # List all objects in the source prefix with pagination
        all_objects = []
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=source_prefix)

        for page in page_iterator:
            if 'Contents' not in page:
                logger.info(f"No files found in {source_prefix}")
                return {
                    'statusCode': 200,
                    'body': f"No files to move in {source_prefix}"
                }
            all_objects.extend(page['Contents'])

        logger.info(f"Found {len(all_objects)} objects in {source_prefix}")

        # Move each file
        for obj in all_objects:
            source_key = obj['Key']
            # Skip if the key is just the prefix itself (e.g., a folder marker)
            if source_key == source_prefix:
                continue

            # Construct the destination key
            relative_key = source_key[len(source_prefix):]
            destination_key = f"{destination_prefix}{relative_key}"

            logger.info(f"Moving {source_key} to {destination_key}")

            # Copy the object to the destination
            copy_source = {'Bucket': bucket, 'Key': source_key}
            s3_client.copy_object(
                Bucket=bucket,
                Key=destination_key,
                CopySource=copy_source
            )

            # Delete the source object
            s3_client.delete_object(Bucket=bucket, Key=source_key)

        logger.info(f"Successfully moved {len(all_objects)} files from {source_prefix} to {destination_prefix}")
        return {
            'statusCode': 200,
            'body': f"Successfully moved {len(all_objects)} files from {source_prefix} to {destination_prefix}"
        }

    except Exception as e:
        logger.error(f"Error moving files: {str(e)}")
        raise e