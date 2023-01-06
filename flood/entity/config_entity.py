from dataclasses import dataclass
from flood.constants import *
import os 

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME:str = BUCKET_NAME
        self.ZIP_FILE_NAME: str = ZIP_FILE_NAME
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,self.ZIP_FILE_NAME)
        self.IMAGE_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,IMAGE)
        self.MASK_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,MASK)
        self.METADETA_ARTIFACTS: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,METADATA)