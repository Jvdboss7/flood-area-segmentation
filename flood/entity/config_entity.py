from dataclasses import dataclass
from flood.constants import *
import os 

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME:str = BUCKET_NAME
        self.ZIP_FILE_NAME: str = ZIP_FILE_NAME
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_INGESTION_ARTIFACTS_DIR + "/" )
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,self.ZIP_FILE_NAME)
        self.IMAGE_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR + "/" + IMAGE)
        self.MASK_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR + "/" + MASK)
        self.METADETA_ARTIFACTS: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR + "/" + METADATA)

@dataclass
class ModelTrainerConfig: 
    def __init__(self):
        self.MODEL_TRAINER_ARTIFACT_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR) 
        self.TRAINED_MODEL_PATH = os.path.join(self.MODEL_TRAINER_ARTIFACT_DIR,TRAINED_MODEL_DIR)

@dataclass
class ModelEvaluationConfig: 
    def __init__(self):
        self.MODEL_EVALUATION_ARTIFACT_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
        self.BEST_MODEL_DIR_PATH: str = os.path.join(self.MODEL_EVALUATION_ARTIFACT_DIR,BEST_MODEL_DIR)
        self.BUCKET_NAME = BUCKET_NAME 
        self.MODEL_DIR = MODEL_DIR
    
@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.TRAINED_MODEL_PATH = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.BUCKET_NAME = BUCKET_NAME
        self.MODEL_DIR = MODEL_DIR