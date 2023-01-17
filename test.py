
import matplotlib.pyplot as plt

import cv2
import tensorflow as tf
import requests, io
import numpy as np
from PIL import Image
from flood.utils.all_utils import create_model
import base64
from io import BytesIO
from numba import cuda
cuda.select_device(0)
cuda.close()




def modify_mask(mask):
    mask = np.expand_dims(mask, axis = 2)
    t_mask = np.zeros(mask.shape)
    np.place(t_mask[:, :, 0], mask[:, :, 0] >=0.5, 1)
    return t_mask

def make_pred_good(pred):
    pred = pred[0][:, :, :]
    pred = modify_mask(pred[:, :, 0])
    pred = np.repeat(pred, 3, 2)
    return pred

def placeMaskOnImg(img, mask):
    np.place(img[:, :, :], mask[:, :, :] >= 0.5, color)
    return img

if __name__ == "__main__":
    
    color = np.array([161, 205, 255])/255.0

    url ='https://raw.githubusercontent.com/jaydeepIneuron007/Dataset/main/0.jpg'

    response = requests.get(url)
    bytes_im = io.BytesIO(response.content)
    img = np.array(Image.open(bytes_im).convert('RGB'))[:, :,:3]

    img = img/255.0
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img, axis = 0)

    tf.keras.backend.clear_session()
    cuda.select_device(0)
    cuda.close()
    model_path = "/home/jvdboss/workspace/ML_DL/flood_area_segmentation/flood-area-segmentation/artifacts/model"

    model = tf.keras.models.load_model(model_path)

    pred = make_pred_good(model(img))

    final_pred = placeMaskOnImg(img[0], pred)

    plt.imshow(final_pred)
    plt.show()

    # PIL_image = Image.fromarray(np.uint8(final_pred)).convert('RGB')

    # buffered = BytesIO()
    # PIL_image.save(buffered, format="JPEG")
    # img_str = base64.b64encode(buffered.getvalue())

    # print(img_str)




