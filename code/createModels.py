import numpy as np
import os
import cv2
import random
import pickle

file_list = []
class_list = []

data_dir = "../data/Datasets/"

class_names = ["fingerLeft", "fingerRight", "fist", "none", "palm", "thumb"]

img_size = 50

for class_name in class_names:
   path = os.path.join(data_dir, class_name)
   for img in os.listdir(path):
       img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

trainImages = []

def createTrainingData():
    for class_name in class_names:
        path = os.path.join(data_dir, class_name)
        class_id = class_names.index(class_name)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resize_array = cv2.resize(img_array, (img_size, img_size))
                trainImages.append([resize_array, class_id])
            except Exception as e:
                pass

createTrainingData()

random.shuffle(trainImages)

x_trains = []
y_labels = []

for feature, label in trainImages:
    x_trains.append(feature)
    y_labels.append(label)


x_trains = np.array(x_trains).reshape(-1, img_size, img_size, 1)
y_labels = np.array(y_labels)

pickle_out = open("../data/Datasets/trainModelImages", "wb")
pickle.dump(x_trains, pickle_out)
pickle_out.close()

pickle_out = open("../data/Datasets/trainModelLabels", "wb")
pickle.dump(y_labels, pickle_out)
pickle_out.close()

pickle_in = open("../data/Datasets/trainModelImages", "rb")
X = pickle.load(pickle_in)
