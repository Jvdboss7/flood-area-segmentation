import sys
import yaml 
from flood.constants import *
from flood.logger import logging 
from flood.exception import CustomException
import segmentation_models as sm




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

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys) from e

