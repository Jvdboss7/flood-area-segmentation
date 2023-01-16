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


IMG_SIZE = (224,224)
BATCH_SIZE = 8
# TRAIN_FILE_NAME = "train_dataset"
# TEST_FILE_NAME = "test_dataset"

# Model training constants
MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifacts'
TRAINED_MODEL_DIR = 'trained_model'
TRAINED_MODEL_NAME = 'model.h5'
TEST_DATASET: str = 'kerasNativeTestData'
# Model  Evaluation constants
MODEL_EVALUATION_ARTIFACTS_DIR = 'ModelEvaluationArtifacts'
BEST_MODEL_DIR = "best_Model"
MODEL_EVALUATION_FILE_NAME = 'loss.csv'
MODEL_NAME = "model.h5"