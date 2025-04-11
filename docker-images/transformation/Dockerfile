# Use python:3.9-slim as the base image
FROM python:3.9-slim

# Install Java, curl, and procps
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk \
    curl \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PySpark and boto3
RUN pip install pyspark==3.4.4 boto3

# Download Hadoop AWS and AWS SDK JARs
RUN mkdir -p /app/jars
RUN curl -o /app/jars/hadoop-aws-3.3.4.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar
RUN curl -o /app/jars/aws-java-sdk-bundle-1.12.767.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.767/aws-java-sdk-bundle-1.12.767.jar

# Set PySpark environment
ENV PYSPARK_PYTHON=python3

# Copy the transformation script
COPY transform.py /app/transform.py

# Set working directory
WORKDIR /app

# Command to run the transformation script
CMD ["python3", "transform.py"]