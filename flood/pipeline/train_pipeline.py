import sys
from flood.logger import logging
from flood.exception import CustomException
from flood.components.data_ingestion import DataIngestion
from flood.components.model_trainer import ModelTrainer
from flood.components.model_evaluation import ModelEvaluation
from flood.components.model_pusher import ModelPusher
from flood.entity.config_entity import DataIngestionConfig, ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig
from flood.entity.artifact_entity import DataIngestionArtifacts, ModelTrainerArtifacts,ModelEvaluationArtifacts,ModelPusherArtifacts
# from flood.utils.all_utils import utils
class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config =ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

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

    def start_model_evaluation(self, model_trainer_artifacts: ModelTrainerArtifacts) -> ModelEvaluationArtifacts:
        logging.info("Entered the start_model_evaluation method of TrainPipeline class")
        try:
            model_evaluation = ModelEvaluation(model_evaluation_config=self.model_evaluation_config,
                                               model_trainer_artifacts=model_trainer_artifacts)

            model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
            logging.info("Exited the start_model_evaluation method of TrainPipeline class")
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_pusher(self,) -> ModelPusherArtifacts:
        logging.info("Entered the start_model_pusher method of TrainPipeline class")
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Initiated the model pusher")
            logging.info("Exited the start_model_pusher method of TrainPipeline class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e


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
            model_evaluation_artifacts = self.start_model_evaluation(model_trainer_artifacts=model_trainer_artifacts
            ) 

            if not model_evaluation_artifacts.is_model_accepted:
                raise Exception("Trained model is not better than the best model")

            model_pusher_artifacts = self.start_model_pusher()

            logging.info("Exited the run_pipeline method of TrainPipeline class")    
         
        except Exception as e:
            raise CustomException(e, sys) from e