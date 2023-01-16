import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flood.logger import logging
from flood.exception import CustomException
from keras.utils import pad_sequences
from flood.constants import *
from flood.utils.all_utils import utils 
import segmentation_models as sm
from flood.configuration.gcloud_syncer import GCloudSync
from sklearn.metrics import confusion_matrix
from flood.entity.config_entity import ModelEvaluationConfig
from flood.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts



class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts):

        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.utils = utils()
        self.gcloud = GCloudSync()

    def get_best_model_from_gcloud(self) -> str:
        """
        :return: Fetch best model from gcloud storage and store inside best model directory path
        """
        try:
            logging.info("Entered the get_best_model_from_gcloud method of Model Evaluation class")

            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)

            self.gcloud.sync_folder_from_gcloud(self.model_evaluation_config.BUCKET_NAME,
                                                self.model_evaluation_config.MODEL_NAME,
                                                self.model_evaluation_config.BEST_MODEL_DIR_PATH)

            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                                           self.model_evaluation_config.MODEL_NAME)
            logging.info("Exited the get_best_model_from_gcloud method of Model Evaluation class")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e 



    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
                Method Name :   initiate_model_evaluation
                Description :   This function is used to initiate all steps of the model evaluation

                Output      :   Returns model evaluation artifact
                On Failure  :   Write an exception log and then raise an exception
        """

        try:
            
            #tokenized_datasets = load_from_disk(self.data_transformation_artifacts.path_tokenized_data)            
            # model = self.model_trainer_artifacts.trained_model_path

            model = self.utils.create_model()
            # model= self.model_trainer.create_model()
            model.compile(
                optimizer = keras.optimizers.Adam(learning_rate = 2e-3),
                loss = keras.losses.BinaryCrossentropy(),
                metrics = [sm.metrics.iou_score],
            )
            test_data = self.model_trainer_artifacts.test_dataset
            print(test_data)
            loss = model.evaluate(test_data)
            print(loss)

            os.makedirs(self.model_evaluation_config.MODEL_EVALUATION_MODEL_DIR, exist_ok=True)
            
            #loss.to_csv(self.model_evaluation_config.EVALUATED_LOSS_CSV_PATH, index=False)

            gcloud_model = self.get_best_model_from_gcloud()
            logging.info(f"{gcloud_model}")

            is_model_accepted = False
            gcloud_loss = None 
            print(f"--------------------------{gcloud_model}--------------------------------")
            #print(f"{os.path.isfile(s3_model)}")
            if gcloud_model is False: 
                is_model_accepted = True
                print("s3 model is false and model accepted is true")
                gcloud_loss = None

            else:
                print("Entered inside the else condition")


                print("Model loaded from gcloud")
                gcloud_loss = model.evaluate(test_data)

                if gcloud_loss > loss:
                    print(f"printing the loss inside the if condition{gcloud_loss} and {loss}")
                    # 0.03 > 0.02
                    is_model_accepted = True
                    print("f{is_model_accepted}")
            model_evaluation_artifact = ModelEvaluationArtifacts(
                        is_model_accepted=is_model_accepted)
            print(f"{model_evaluation_artifact}")

            logging.info("Exited the initiate_model_evaluation method of Model Evaluation class")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
