import os
import sys
import logging
import boto3
from decimal import Decimal
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, TimestampType
from pyspark.sql.functions import col, to_date, sum as _sum, avg, count, countDistinct, when, round

# Set up logging to CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# Get environment variables
S3_BUCKET = os.getenv('S3_BUCKET')
S3_FILE_PRODUCTS = os.getenv('S3_FILE_PRODUCTS')
S3_FILE_ORDERS = os.getenv('S3_FILE_ORDERS')
S3_FILE_ORDER_ITEMS = os.getenv('S3_FILE_ORDER_ITEMS')
DYNAMODB_CATEGORY_TABLE = os.getenv('DYNAMODB_CATEGORY_TABLE')
DYNAMODB_ORDER_TABLE = os.getenv('DYNAMODB_ORDER_TABLE')
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

# Define schemas (same as validate.py)
products_schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("sku", StringType(), False),
    StructField("cost", FloatType(), False),
    StructField("category", StringType(), False),
    StructField("name", StringType(), False),
    StructField("brand", StringType(), True),
    StructField("retail_price", FloatType(), False),
    StructField("department", StringType(), False)
])

orders_schema = StructType([
    StructField("order_id", IntegerType(), False),
    StructField("user_id", IntegerType(), False),
    StructField("status", StringType(), False),
    StructField("created_at", TimestampType(), False),
    StructField("returned_at", TimestampType(), True),
    StructField("shipped_at", TimestampType(), True),
    StructField("delivered_at", TimestampType(), True),
    StructField("num_of_item", IntegerType(), False)
])

order_items_schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("order_id", IntegerType(), False),
    StructField("user_id", IntegerType(), False),
    StructField("product_id", IntegerType(), False),
    StructField("status", StringType(), False),
    StructField("created_at", TimestampType(), False),
    StructField("shipped_at", TimestampType(), True),
    StructField("delivered_at", TimestampType(), True),
    StructField("returned_at", TimestampType(), True),
    StructField("sale_price", FloatType(), False)
])

# Initialize Spark session with S3A configurations
spark = SparkSession.builder \
    .appName("e_commerce_kpi_transformation") \
    .config("spark.jars", "/app/jars/hadoop-aws-3.3.4.jar,/app/jars/aws-java-sdk-bundle-1.12.767.jar") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

def load_data():
    try:
        # Construct S3 paths with s3a://
        products_path = f"s3a://{S3_BUCKET}/{S3_FILE_PRODUCTS}"
        orders_path = f"s3a://{S3_BUCKET}/{S3_FILE_ORDERS}/"
        order_items_path = f"s3a://{S3_BUCKET}/{S3_FILE_ORDER_ITEMS}/"

        logger.info(f"Loading data from S3: {products_path}, {orders_path}, {order_items_path}")
        products_df = spark.read.schema(products_schema).csv(products_path, header=True)
        orders_df = spark.read.schema(orders_schema).csv(orders_path, header=True)
        order_items_df = spark.read.schema(order_items_schema).csv(order_items_path, header=True)

        # Cache the DataFrames for performance
        products_df.cache()
        orders_df.cache()
        order_items_df.cache()

    except Exception as e:
        logger.error(f"Error loading data from S3: {str(e)}")
        raise e

    return products_df, orders_df, order_items_df

def compute_category_kpis(products_df, orders_df, order_items_df):
    logger.info("Computing Category-Level KPIs...")
    # Join order_items with orders and products
    order_items_with_orders = order_items_df.alias("oi").join(
        orders_df.alias("o"),
        col("oi.order_id") == col("o.order_id"),
        "inner"
    )
    order_items_with_products = order_items_with_orders.join(
        products_df.alias("p"),
        col("oi.product_id") == col("p.id"),
        "inner"
    )

    # Extracting the order date (without time) and computing KPIs Category-Level KPIs
    category_kpis = (order_items_with_products
        .withColumn("order_date", to_date(col("oi.created_at")))
        .groupBy("p.category", "order_date")
        .agg(
            round(_sum("oi.sale_price"), 2).alias("daily_revenue"),
            round(avg("oi.sale_price"), 2).alias("avg_order_value"),
            round(
                (count(when(col("oi.status") == "returned", 1)) / count("*")) * 100, 2
            ).alias("avg_return_rate")
        ).orderBy("category", "order_date")
    )

    return category_kpis

def compute_order_kpis(orders_df, order_items_df):
    logger.info("Computing Order-Level KPIs...")
    # Join orders with order_items
    orders_with_items = orders_df.alias("o").join(
        order_items_df.alias("oi"),
        col("o.order_id") == col("oi.order_id"),
        "inner"
    )

    # Extracting the order date (without time) and computing Order-Level KPIs
    order_kpis = (orders_with_items
        .withColumn("order_date", to_date(col("o.created_at")))
        .groupBy("order_date")
        .agg(
            countDistinct("o.order_id").alias("total_orders"),
            round(_sum("oi.sale_price"), 2).alias("total_revenue"),
            _sum("o.num_of_item").alias("total_items_sold"),
            round(
                (count(when(col("o.status") == "returned", 1)) / countDistinct("o.order_id")) * 100, 2
            ).alias("return_rate"),
            countDistinct("o.user_id").alias("unique_customers")
        ).orderBy("order_date")
    )

    return order_kpis

def store_category_kpis_in_dynamodb(category_kpis_df, table_name):
    try:
        logger.info(f"Storing category KPIs in DynamoDB table: {table_name}")
        dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
        table = dynamodb.Table(table_name)
        logger.info(f"Attempting to write to DynamoDB table: {table_name}")

        collected_rows = category_kpis_df.collect()
        logger.info(f"Collected {len(collected_rows)} rows for category KPIs.")

        with table.batch_writer() as batch:
            for row in collected_rows:
                item = {
                    'category': row['category'],
                    'order_date': row['order_date'].strftime('%Y-%m-%d'),
                    'daily_revenue': Decimal(str(row['daily_revenue'])) if row['daily_revenue'] is not None else Decimal('0'),
                    'avg_order_value': Decimal(str(row['avg_order_value'])) if row['avg_order_value'] is not None else Decimal('0'),
                    'avg_return_rate': Decimal(str(row['avg_return_rate'])) if row['avg_return_rate'] is not None else Decimal('0')
                }
                logger.debug(f"Putting category item: {item}")
                batch.put_item(Item=item)

        logger.info(f"Category KPIs successfully stored in DynamoDB table: {table_name}")
    except Exception as e:
        logger.error(f"Error storing category KPIs in DynamoDB: {e}")
        raise

def store_order_kpis_in_dynamodb(order_kpis_df, table_name):
    try:
        logger.info(f"Storing order KPIs in DynamoDB table: {table_name}")
        dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
        table = dynamodb.Table(table_name)
        logger.info(f"Attempting to write to DynamoDB table: {table_name}")

        collected_rows = order_kpis_df.collect()
        logger.info(f"Collected {len(collected_rows)} rows for order KPIs.")

        with table.batch_writer() as batch:
            for row in collected_rows:
                item = {
                    'order_date': row['order_date'].strftime('%Y-%m-%d'),
                    'total_orders': int(row['total_orders']) if row['total_orders'] is not None else 0,
                    'total_revenue': Decimal(str(row['total_revenue'])) if row['total_revenue'] is not None else Decimal('0'),
                    'total_items_sold': int(row['total_items_sold']) if row['total_items_sold'] is not None else 0,
                    'return_rate': Decimal(str(row['return_rate'])) if row['return_rate'] is not None else Decimal('0'),
                    'unique_customers': int(row['unique_customers']) if row['unique_customers'] is not None else 0
                }
                logger.debug(f"Putting order item: {item}")
                batch.put_item(Item=item)

        logger.info(f"Order KPIs successfully stored in DynamoDB table: {table_name}")
    except Exception as e:
        logger.error(f"Error storing order KPIs in DynamoDB: {e}")
        raise

def main():
    try:
        # Load data
        products_df, orders_df, order_items_df = load_data()

        # Compute KPIs
        category_kpis = compute_category_kpis(products_df, orders_df, order_items_df)
        order_kpis = compute_order_kpis(orders_df, order_items_df)

        # Store KPIs in DynamoDB
        store_category_kpis_in_dynamodb(category_kpis, DYNAMODB_CATEGORY_TABLE)
        store_order_kpis_in_dynamodb(order_kpis, DYNAMODB_ORDER_TABLE)

        logger.info("Transformation and storage completed successfully.")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Transformation task failed with error: {str(e)}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()