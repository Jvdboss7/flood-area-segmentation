import os
import re
import sys
import cv2 as op
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
from flood.constants import *
import matplotlib.pyplot as plt
from flood.logger import logging 
from flood.exception import CustomException
from sklearn.model_selection import train_test_split
from flood.entity.config_entity import DataTransformationConfig
from flood.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts

class DataTransformation:
    def __init__(self,data_transformation_config: DataTransformationConfig,data_ingestion_artifacts:DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    def load_data(self):

        try:
            root_dir = self.data_ingestion_artifacts.root_dir
            df = pd.read_csv(self.data_ingestion_artifacts.metadata_file_path)

            df['Image'] = df['Image'].map(lambda x: root_dir + 'Image/' + x)
            df['Mask'] = df['Mask'].map(lambda x: root_dir + 'Mask/' + x)
            return df
        except Exception as e:
            raise CustomException(e,sys) from e
    def transform_data(self):
        
        try:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Blur(blur_limit = 3, p = 0.5),
                # A.RandomRotate90(p=1) ,
                # A.Rotate(limit=90,p=0.5) ,
                # A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
                
            ])
            return transform
        except Exception as e:
            raise CustomException(e,sys) from e

    # The below function is used to change the dimensions of the masked images
    def modify_mask(self,mask):
        try:
            mask = np.expand_dims(mask, axis = 2)
            t_mask = np.zeros(mask.shape)
            np.place(t_mask[:, :, 0], mask[:, :, 0] >=0.5, 1)
            return t_mask
        except Exception as e:
            raise CustomException(e,sys) from e

    # The below function is used to map the images with masked images
    def map_function(self,img, mask, training):
        try:
            img, mask = plt.imread(img.decode())[:, :, :3], plt.imread(mask.decode())
            img = op.resize(img, IMG_SIZE)
            mask = self.modify_mask(op.resize(mask, IMG_SIZE))
            
            img = img/255.0
            if training == True:
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']
            
            return img.astype(np.float64), mask.astype(np.float64)
        except Exception as e:
            raise CustomException(e,sys) from e
    
    def create_dataset(self,data, training = True):
        try:
            dataset = tf.data.Dataset.from_tensor_slices((data['Image'], data['Mask']))
            dataset = dataset.shuffle(100)
            dataset = dataset.map(lambda img, mask : tf.numpy_function(
                            self.map_function, [img, mask, training], [tf.float64, tf.float64]),
                            num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)

            dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
            return dataset
        except Exception as e:
            raise CustomException(e,sys) from e

    def split_data(self,df):
        try:
            df_train, df_test = train_test_split(df, test_size = 0.3)

            print(df_train.shape, df_test.shape)

            return df_train,df_test
        except Exception as e:
            raise CustomException(e,sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")

            df = self.load_data()
            # transform = self.transform_data()
            # t_mask = self.modify_mask()
            # dataset = self.create_dataset(df)
            df_train,df_test = self.split_data(df)

            train_dataset = self.create_dataset(df_train, training = True)
            test_dataset = self.create_dataset(df_test, training = False)
            print(f"========={type(train_dataset)}===========")
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            #df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH,index=False,header=True)

            data_transformation_artifact = DataTransformationArtifacts(
                train_data_path = self.data_transformation_config.TRAIN_FILE_PATH,
                test_data_path = self.data_transformation_config.TEST_FILE_PATH
            )            
        except Exception as e:
            raise CustomException(e, sys) from e