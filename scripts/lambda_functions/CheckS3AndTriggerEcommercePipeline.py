import boto3
import os
import logging
import json
import urllib.parse

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
S3_BUCKET = os.getenv('S3_BUCKET', 'ecommerce-transactional-bucket')
S3_PRODUCTS_PATH = os.getenv('S3_PRODUCTS_PATH', 'landing-data/products.csv')
S3_ORDERS_PATH = os.getenv('S3_ORDERS_PATH', 'landing-data/orders/')
S3_ORDER_ITEMS_PATH = os.getenv('S3_ORDER_ITEMS_PATH', 'landing-data/order_items/')
STATE_MACHINE_ARN = os.getenv('STATE_MACHINE_ARN', 'arn:aws:states:eu-west-1:195275667627:stateMachine:EcommercePipeline')

# Initialize S3 and Step Functions clients
s3_client = boto3.client('s3')
sfn_client = boto3.client('stepfunctions')

def check_s3_files():
    """
    Check if all required files/folders are present in S3.
    Returns True if all files are present, False otherwise.
    """
    try:
        # Check for products.csv
        logger.info(f"Checking for {S3_PRODUCTS_PATH} in S3 bucket {S3_BUCKET}")
        s3_client.head_object(Bucket=S3_BUCKET, Key=S3_PRODUCTS_PATH)
        logger.info(f"Found {S3_PRODUCTS_PATH}")

        # Check for at least one file in the orders folder
        logger.info(f"Checking for files in {S3_ORDERS_PATH}")
        orders_response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_ORDERS_PATH)
        orders_files = [obj['Key'] for obj in orders_response.get('Contents', []) if obj['Key'] != S3_ORDERS_PATH]
        if not orders_files:
            logger.warning(f"No files found in {S3_ORDERS_PATH}")
            return False
        logger.info(f"Found {len(orders_files)} files in {S3_ORDERS_PATH}")

        # Check for at least one file in the order_items folder
        logger.info(f"Checking for files in {S3_ORDER_ITEMS_PATH}")
        order_items_response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_ORDER_ITEMS_PATH)
        order_items_files = [obj['Key'] for obj in order_items_response.get('Contents', []) if obj['Key'] != S3_ORDER_ITEMS_PATH]
        if not order_items_files:
            logger.warning(f"No files found in {S3_ORDER_ITEMS_PATH}")
            return False
        logger.info(f"Found {len(order_items_files)} files in {S3_ORDER_ITEMS_PATH}")

        return True

    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.warning(f"File {S3_PRODUCTS_PATH} not found in S3 bucket {S3_BUCKET}")
        else:
            logger.error(f"Error checking S3 files: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking S3 files: {str(e)}")
        return False

def trigger_step_functions():
    """
    Trigger the Step Functions state machine.
    """
    try:
        # Check if the state machine is already running
        response = sfn_client.list_executions(
            stateMachineArn=STATE_MACHINE_ARN,
            statusFilter='RUNNING'
        )
        if response['executions']:
            logger.info("Step Functions state machine is already running. Skipping execution.")
            return {
                'statusCode': 200,
                'body': "Step Functions state machine is already running. Execution not triggered."
            }

        logger.info("All required files are present. Triggering Step Functions state machine.")
        response = sfn_client.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            input='{}'  # No input needed for this pipeline
        )
        logger.info(f"Started Step Functions execution: {response['executionArn']}")
        return {
            'statusCode': 200,
            'body': f"Started Step Functions execution: {response['executionArn']}"
        }
    except Exception as e:
        logger.error(f"Error triggering Step Functions: {str(e)}")
        raise e

def lambda_handler(event, context):
    """
    Lambda handler to process S3 events from SQS.
    """
    try:
        # Check if the event is from SQS (S3 event notification)
        if 'Records' not in event or not event['Records']:
            logger.warning("No SQS records found in event.")
            return {
                'statusCode': 400,
                'body': "Invalid event format. Expected SQS records."
            }

        for record in event['Records']:
            # Parse the SQS message
            body = json.loads(record['body'])
            if 'Records' not in body:
                logger.warning(f"Unexpected SQS message format: {body}")
                return {
                    'statusCode': 400,
                    'body': "Invalid SQS message format."
                }

            for s3_record in body['Records']:
                bucket = s3_record['s3']['bucket']['name']
                key = urllib.parse.unquote_plus(s3_record['s3']['object']['key'])
                logger.info(f"Received S3 event for bucket {bucket}, key {key}")

                # Check if all required files are present
                if check_s3_files():
                    return trigger_step_functions()
                else:
                    logger.info("Not all required files are present in S3 after S3 event. Waiting for more files.")
                    return {
                        'statusCode': 200,
                        'body': "Required files not yet complete in S3. Step Functions execution not triggered."
                    }

    except Exception as e:
        logger.error(f"Error in Lambda handler: {str(e)}")
        raise e