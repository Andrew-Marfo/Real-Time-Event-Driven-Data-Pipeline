import os
import sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, TimestampType

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
S3_FILE_ORDERS = os.getenv('S3_FILE_ORDERS')  # Now points to the folder (e.g., landing-data/orders)
S3_FILE_ORDER_ITEMS = os.getenv('S3_FILE_ORDER_ITEMS')  # Now points to the folder (e.g., landing-data/order_items)

# Define schemas
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

# Initialize Spark session
spark = SparkSession.builder.appName("e_commerce_validation").getOrCreate()

def load_data():
    # Construct S3 paths
    products_path = f"s3://{S3_BUCKET}/{S3_FILE_PRODUCTS}"
    orders_folder_path = f"s3://{S3_BUCKET}/{S3_FILE_ORDERS}/"  # Folder path
    order_items_folder_path = f"s3://{S3_BUCKET}/{S3_FILE_ORDER_ITEMS}/"  # Folder path

    logger.info(f"Loading data from S3:")
    logger.info(f"Products: {products_path}")
    logger.info(f"Orders (all files in folder): {orders_folder_path}")
    logger.info(f"Order Items (all files in folder): {order_items_folder_path}")

    try:
        # Read the single products.csv file
        products_df = spark.read.schema(products_schema).csv(products_path, header=True)

        # Read all CSV files in the orders folder
        orders_df = spark.read.schema(orders_schema).csv(orders_folder_path, header=True)

        # Read all CSV files in the order_items folder
        order_items_df = spark.read.schema(order_items_schema).csv(order_items_folder_path, header=True)

        # Check if any DataFrame is empty
        if products_df.count() == 0:
            raise ValueError("Products DataFrame is empty. No data found in products.csv.")
        if orders_df.count() == 0:
            raise ValueError("Orders DataFrame is empty. No data found in orders folder.")
        if order_items_df.count() == 0:
            raise ValueError("Order Items DataFrame is empty. No data found in order_items folder.")

    except Exception as e:
        logger.error(f"Failed to load data from S3: {str(e)}")
        raise e

    return products_df, orders_df, order_items_df

def validate_data():
    # Load data
    products_df, orders_df, order_items_df = load_data()

    # Validation 1: Check for missing fields (non-nullable columns)
    logger.info("Checking for missing fields...")
    
    # Products: All fields are non-nullable
    for field in products_schema.fields:
        if not field.nullable:
            null_count = products_df.filter(products_df[field.name].isNull()).count()
            if null_count > 0:
                logger.error(f"Missing values found in products.csv for field: {field.name}")
                return False

    # Orders: Non-nullable fields are order_id, user_id, status, created_at, num_of_item
    non_nullable_order_fields = [f.name for f in orders_schema.fields if not f.nullable]
    for field in non_nullable_order_fields:
        null_count = orders_df.filter(orders_df[field].isNull()).count()
        if null_count > 0:
            logger.error(f"Missing values found in orders data for field: {field}")
            return False

    # Order Items: Non-nullable fields are id, order_id, user_id, product_id, status, created_at, sale_price
    non_nullable_order_item_fields = [f.name for f in order_items_schema.fields if not f.nullable]
    for field in non_nullable_order_item_fields:
        null_count = order_items_df.filter(order_items_df[field].isNull()).count()
        if null_count > 0:
            logger.error(f"Missing values found in order_items data for field: {field}")
            return False

    # Validation 2: Check referential integrity
    logger.info("Checking referential integrity...")
    
    # Ensure product_id in order_items exists in products
    invalid_products = order_items_df.join(
        products_df,
        order_items_df.product_id == products_df.id,
        "left_anti"
    )
    invalid_product_count = invalid_products.count()
    if invalid_product_count > 0:
        invalid_product_ids = invalid_products.select("product_id").distinct().collect()
        logger.error(f"Invalid product_ids found in order_items data: {[row['product_id'] for row in invalid_product_ids]}")
        return False

    # Ensure order_id in order_items exists in orders
    invalid_orders = order_items_df.join(
        orders_df,
        order_items_df.order_id == orders_df.order_id,
        "left_anti"
    )
    invalid_order_count = invalid_orders.count()
    if invalid_order_count > 0:
        invalid_order_ids = invalid_orders.select("order_id").distinct().collect()
        logger.error(f"Invalid order_ids found in order_items data: {[row['order_id'] for row in invalid_order_ids]}")
        return False

    # Validation 3: Check for duplicate order_ids in orders
    logger.info("Checking for duplicate order_ids in orders...")
    duplicate_orders = orders_df.groupBy("order_id").count().filter("count > 1")
    if duplicate_orders.count() > 0:
        duplicate_order_ids = duplicate_orders.select("order_id").collect()
        logger.error(f"Duplicate order_ids found in orders data: {[row['order_id'] for row in duplicate_order_ids]}")
        return False

    # Validation 4: Check for duplicate ids in order_items
    logger.info("Checking for duplicate ids in order_items...")
    duplicate_order_items = order_items_df.groupBy("id").count().filter("count > 1")
    if duplicate_order_items.count() > 0:
        duplicate_ids = duplicate_order_items.select("id").collect()
        logger.error(f"Duplicate ids found in order_items data: {[row['id'] for row in duplicate_ids]}")
        return False

    logger.info("All validations passed!")
    return True

if __name__ == "__main__":
    try:
        if validate_data():
            logger.info("Validation successful. Data is ready for transformation.")
            sys.exit(0)
        else:
            logger.error("Validation failed. Check logs for details.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Validation task failed with error: {str(e)}")
        sys.exit(1)
    finally:
        spark.stop()