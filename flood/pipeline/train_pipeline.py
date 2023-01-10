import sys
from flood.logger import logging
from flood.exception import CustomException
from flood.components.data_ingestion import DataIngestion
from flood.components.model_trainer import ModelTrainer
from flood.entity.config_entity import DataIngestionConfig, ModelTrainerConfig
from flood.entity.artifact_entity import DataIngestionArtifacts, ModelTrainerArtifacts
# from flood.utils.all_utils import utils
class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.model_trainer_config = ModelTrainerConfig()
        # self.utils = utils
    def start_data_ingestion(self) -> DataIngestionArtifacts:

        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Getting the data from GCLoud Storage bucket")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)

            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train and valid from GCLoud Storage")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e


    def start_model_trainer(self,data_ingestion_artifacts:DataIngestionArtifacts)->ModelTrainerArtifacts:
        logging.info(
            "Entered the start_model_trainer method of TrainPipeline class"
        )
        try:
            model_trainer = ModelTrainer(data_ingestion_artifacts =data_ingestion_artifacts,
                                        model_trainer_config=self.model_trainer_config)
            model_trainer_artifacts = model_trainer.initiate_model_trainer()
            logging.info("Exited the start_model_trainer method of the TrainPipeline class")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e,sys) from e

    def run_pipeline(self) -> None:
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()

            # data_transformation_artifacts = self.start_data_transformation(
            #     data_ingestion_artifacts=data_ingestion_artifacts
            # )
            model_trainer_artifacts = self.start_model_trainer(
                data_ingestion_artifacts = data_ingestion_artifacts
            )
         
        except Exception as e:
            raise CustomException(e, sys) from e