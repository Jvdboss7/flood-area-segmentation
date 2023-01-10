import os 
import sys
import cv2 as op
import numpy as np
import albumentations as A
import tensorflow as tf
import matplotlib.pyplot as plt
from flood.constants import *
from flood.exception import CustomException

class utils:
    def __init__(self):
        pass

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
                transformed = self.transform_data(image=img, mask=mask)
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

            # dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
            return dataset
        except Exception as e:
            raise CustomException(e,sys) from e