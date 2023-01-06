import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME = 'flood-area'
ZIP_FILE_NAME = 'datasets.zip'

# Data ingestion constants 

DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
IMAGE = "Image"
MASK = "Mask"
METADATA = "metadata.csv"