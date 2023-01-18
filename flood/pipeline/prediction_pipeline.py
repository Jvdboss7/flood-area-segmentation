import sys
import base64
import numpy as np
import requests, io
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import tensorflow as tf
from flood.constants import *
from flood.utils.all_utils import create_model
from PIL import Image as Img
from flood.logger import logging
from flood.exception import CustomException
from flood.utils.all_utils import read_yaml_file
from flood.configuration.gcloud_syncer import GCloudSync


class PredictionPipeline:
    def __init__(self,):
        self.bucket_name = BUCKET_NAME
        self.model_dir = MODEL_DIR
        self.gcloud = GCloudSync()

    def image_loader(self, image_bytes):
        """
        Method Name :   image_loader
        Description :   This method load byte image and save it to local.

        Output      :   Returns path the of the saved image
        """
        logging.info("Entered the image_loader method of PredictionPipeline class")
        try:
            logging.info("load byte image and save it to local")
            image = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))[:, :,:3]

            image = image/255.0
            image = cv2.resize(image, (224,224))
            image = np.expand_dims(image, axis = 0)
            return image

        except Exception as e:
            raise CustomException(e, sys) from e

    def modify_mask(self,mask):
        try:
            mask = np.expand_dims(mask, axis = 2)
            t_mask = np.zeros(mask.shape)
            np.place(t_mask[:, :, 0], mask[:, :, 0] >=0.5, 1)
            return t_mask
        except Exception as e:
            raise CustomException(e,sys) from e

    def make_pred_good(self,pred):
        try:
            pred = pred[0][:, :, :]
            pred = self.modify_mask(pred[:, :, 0])
            pred = np.repeat(pred, 3, 2)
            return pred
        except Exception as e:
            raise CustomException(e,sys) from e

    def placeMaskOnImg(self,img, mask):
        try:
            color = np.array([161, 205, 255])/255.0
            np.place(img[:, :, :], mask[:, :, :] >= 0.5, color)
            return img
        except Exception as e:
            raise CustomException(e,sys) from e

    def get_model_from_gcloud(self) -> str:
        """
        Method Name :   get_model_from_gcloud
        Description :   This method fetched the best model from the gcloud.

        Output      :   Return best model path
        """
        logging.info("Entered the get_model_from_gcloud method of PredictionPipeline class")
        try:
            logging.info("Loading the best model from gcloud bucket")
            os.makedirs("artifacts/PredictModel", exist_ok=True)
            predict_model_path = os.path.join(os.getcwd(), "artifacts", "PredictModel")
            self.gcloud.sync_model_from_gcloud(self.bucket_name, self.model_dir, predict_model_path)
            best_model_path = os.path.join(predict_model_path, self.model_dir)
            logging.info("Exited the get_model_from_gcloud method of PredictionPipeline class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def prediction(self, best_model_path: str, image) -> float:
        """
        Method Name :   prediction
        Description :   This method takes best model path and image

        Output      :   Return the image in base64
        """
        logging.info("Entered the prediction method of PredictionPipeline class")
        try:            
            model = tf.keras.models.load_model(best_model_path)

            pred = self.make_pred_good(model(image))

            final_pred = self.placeMaskOnImg(image[0], pred)
            # plt.imshow(final_pred)
            # plt.show()

            print("Final Pred", final_pred)


            PIL_image = Image.fromarray((final_pred * 255).astype(np.uint8))

            print("PIL_image", PIL_image)
            PIL_image.save(os.path.join(os.getcwd(), 'prediction.png'))

            # buffered = BytesIO()
            # PIL_image.save(buffered, format="JPEG")
            # img_str = base64.b64encode(buffered.getvalue())
            with open(os.path.join(os.getcwd(), 'prediction.png'), "rb") as f:
                img_str= base64.b64encode(f.read())
                

            print(img_str)
            return img_str

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, data):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            image = self.image_loader(data)
            print(image, type(image))
            best_model_path: str = self.get_model_from_gcloud()
            detected_image = self.prediction(best_model_path, image)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return detected_image
        except Exception as e:
            raise CustomException(e, sys) from e