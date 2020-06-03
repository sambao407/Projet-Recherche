import cv2
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.15,
                             zoom_range=0.1,
                             channel_shift_range=10.)

data_dir = "../data/Datasets/"
class_names = ["fingerLeft", "fingerRight", "fist", "none", "palm", "thumb"]

for class_name in class_names:
   path = os.path.join(data_dir, class_name)
   for img in os.listdir(path):
       npImage = np.expand_dims(cv2.cvtColor(cv2.imread(os.path.join(path, img)), cv2.COLOR_BGR2RGB), 0)

       i = 0
       for batch in datagen.flow(npImage, batch_size=1,
           save_to_dir= path,
           save_prefix="image", save_format="jpg"):
           i+=1
           if i >5:
               break