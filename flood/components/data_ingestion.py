import os
import sys
from zipfile import ZipFile
from flood.logger import logging
from flood.exception import CustomException
from flood.configuration.gcloud_syncer import GCloudSync
from flood.entity.config_entity import DataIngestionConfig
from flood.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):

        """
        :param data_ingestion_config: Configuration for data ingestion
        """
        self.data_ingestion_config = data_ingestion_config
        self.gcloud = GCloudSync()

    def get_data_from_gcloud(self) -> None:
        try:
            logging.info("Entered the get_data_from_gcloud method of Data ingestion class")

            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            self.gcloud.sync_folder_from_gcloud(self.data_ingestion_config.BUCKET_NAME,
                                                self.data_ingestion_config.ZIP_FILE_NAME,
                                                self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,
                                                )

            logging.info("Exited the get_data_from_gcloud method of Data ingestion class")

        except Exception as e:
            raise CustomException(e, sys) from e

    def unzip_and_clean(self):

        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try:
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)

            logging.info("Exited the unzip_and_clean method of Data ingestion class")

            return self.data_ingestion_config.IMAGE_ARTIFACTS_DIR,self.data_ingestion_config.MASK_ARTIFACTS_DIR,self.data_ingestion_config.METADETA_ARTIFACTS

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:

        """
        Method Name :   initiate_data_ingestion
        Description :   This function initiates a data ingestion steps
        Output      :   Returns data ingestion artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")
        try:

            self.get_data_from_gcloud()

            logging.info("Fetched the data from S3 bucket")

            image_data_file_path, masks_data_file_path,metadata_file_path = self.unzip_and_clean()

            logging.info("Unzipped file and split into train and valid")

            data_ingestion_artifacts = DataIngestionArtifacts(root_dir=self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, 
                                                            image_data_file_path=image_data_file_path,
                                                              masks_data_file_path=masks_data_file_path,
                                                              metadata_file_path=metadata_file_path)

            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")

            logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e