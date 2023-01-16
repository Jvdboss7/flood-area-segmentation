import sys 
from flood.constants import *
from flood.logger import logging 
from flood.exception import CustomException
import segmentation_models as sm

class utils:
    def __init__(self):
        pass
    
    @staticmethod
    def create_model():
        try:
            logging.info("Entered the create_model function")
            model = sm.Unet('efficientnetb2', 
                            input_shape = (224,224,3), 
                            classes = 1, 
                            activation='sigmoid', 
                            encoder_weights='imagenet')
            logging.info("Exited the create_model function ")
            return model 
        except Exception as e:
            raise CustomException(e,sys) from e
