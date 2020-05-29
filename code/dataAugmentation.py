import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import os

datagen = ImageDataGenerator(rotation_range=40,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             brightness_range=(0.5, 1.5))

data_dir = "../data/Datasets/"
class_names = ["fingerLeft", "fingerRight", "fist", "none", "palm", "thumb"]

for class_name in class_names:
   path = os.path.join(data_dir, class_name)
   for img in os.listdir(path):
       image = cv2.imread(os.path.join(path, img))

       x=img_to_array(image)
       x=x.reshape((1, ) + x.shape)

       i = 0
       for batch in datagen.flow(x, batch_size=1,
           save_to_dir= os.path.join(data_dir, class_name),
           save_prefix="image", save_format="jpg"):
           i+=1
           if i >5:
               break