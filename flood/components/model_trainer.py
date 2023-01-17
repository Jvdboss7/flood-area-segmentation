import os
import re
import sys
import cv2 as op
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
from tensorflow import keras
from flood.constants import *
from keras import backend as K
import matplotlib.pyplot as plt
import segmentation_models as sm
from flood.logger import logging 
from flood.exception import CustomException
from sklearn.model_selection import train_test_split
from flood.utils.all_utils import create_model
from flood.entity.config_entity import ModelTrainerConfig
from flood.entity.artifact_entity import DataIngestionArtifacts, ModelTrainerArtifacts
from keras.layers import Dense, Dropout, Input, add, Conv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose,Activation, Concatenate

class ModelTrainer:
    def __init__(self,model_trainer_config: ModelTrainerConfig,data_ingestion_artifacts:DataIngestionArtifacts):
        self.model_trainer_config = model_trainer_config
        self.data_ingestion_artifacts = data_ingestion_artifacts
        

    def load_data(self):
        try:
            logging.info("Entered into to the load_data fucntion")
            root_dir = self.data_ingestion_artifacts.root_dir
            df = pd.read_csv(self.data_ingestion_artifacts.metadata_file_path)

            df['Image'] = df['Image'].map(lambda x: root_dir + 'Image/' + x)
            df['Mask'] = df['Mask'].map(lambda x: root_dir + 'Mask/' + x)
            logging.info(f"The data is {df}")
            return df
        except Exception as e:
            raise CustomException(e,sys) from e
    def transform_data(self):
        try:
            logging.info("Entered into the transform_data function")
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Blur(blur_limit = 3, p = 0.5),
                A.RandomRotate90(p=1) ,
                A.Rotate(limit=90,p=0.5) ,
                A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            ])
            logging.info("Exited the transform_data function")
            return transform
        except Exception as e:
            raise CustomException(e,sys) from e

    # The below function is used to change the dimensions of the masked images
    def modify_mask(self,mask):
        try:
            logging.info("Entered the modify mask function")
            mask = np.expand_dims(mask, axis = 2)
            t_mask = np.zeros(mask.shape)
            np.place(t_mask[:, :, 0], mask[:, :, 0] >=0.5, 1)
            logging.info("Exited the modify mask function")
            return t_mask
        except Exception as e:
            raise CustomException(e,sys) from e

    #The below function is used to map the images with masked images
    def map_function(self,img, mask, training):
        try:
            logging.info("Entered the map_function")
            img, mask = plt.imread(img.decode())[:, :, :3], plt.imread(mask.decode())
            img = op.resize(img, IMG_SIZE)
            mask = self.modify_mask(op.resize(mask, IMG_SIZE))
            
            img = img/255.0
            if training == True:
                transform = self.transform_data()
                transformed = transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']
            logging.info("Exited the map_function")
            return img.astype(np.float64), mask.astype(np.float64)
        except Exception as e:
            raise CustomException(e,sys) from e
    
    def create_dataset(self,data, training = True):
        try:
            logging.info("Entered the create_dataset function")
            dataset = tf.data.Dataset.from_tensor_slices((data['Image'], data['Mask']))
            dataset = dataset.shuffle(100)
            dataset = dataset.map(lambda img, mask : tf.numpy_function(
                            self.map_function, [img, mask, training], [tf.float64, tf.float64]),
                            num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)

            dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
            logging.info(f"The dataset is {dataset}")
            logging.info("Exited the create_dataset function")
            return dataset
        except Exception as e:
            raise CustomException(e,sys) from e

    def split_data(self,df):
        try:
            logging.info("Entered the split_data function")
            df_train, df_test = train_test_split(df, test_size = TEST_SIZE)

            print(df_train.shape, df_test.shape)

            logging.info(f"The splitted datais {df_train} and {df_test} ")
            logging.info("Exited the split_data")
            return df_train,df_test
        except Exception as e:
            raise CustomException(e,sys) from e

    def dice_coef(self,y_true, y_pred):
        try:
            logging.info("Entered the dice_coef function")
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            logging.info("Exited dice_coef function")
            return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0) 
        except Exception as e:
            raise CustomException(e,sys) from e

    def dice_coef_loss(self,y_true, y_pred):
        try:
            logging.info("Entered the dice_coef_loss function")
            logging.info(f"The function reutrns {1-self.dice_coef(y_true, y_pred)}")
            return 1-self.dice_coef(y_true, y_pred)
        except Exception as e:
            raise CustomException(e,sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")

            df = self.load_data()
            
            df_train,df_test = self.split_data(df)

            train_dataset = self.create_dataset(df_train, training = True)
            test_dataset = self.create_dataset(df_test, training = False)
            

            print(f"========={type(train_dataset)}===========")
            print(f"========={type(test_dataset)}===========")

            model = create_model()
            # To check the model summary

            model.summary()

            # Compiling the model
            model.compile(
                optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE),
                loss = keras.losses.BinaryCrossentropy()
            )

            history = model.fit(train_dataset, validation_data = test_dataset, epochs = EPOCHS)
            print(f"---------------{history.history}--------------")
            

            os.makedirs(self.model_trainer_config.TRAINED_MODEL_PATH,exist_ok=True)
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            # trained_model_path = self.model_trainer_config.TRAINED_MODEL_PATH
            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_PATH,
                test_dataset=test_dataset)
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts           
        except Exception as e:
            raise CustomException(e, sys) from e