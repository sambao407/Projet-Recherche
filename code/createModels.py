import numpy as np
import os
import cv2
import random
import pickle

file_list = []
class_list = []

#Path of the differents datasets
data_dir = "../data/Datasets/"

#All the classes the neural network has to recognize
class_names = ["fingerLeft", "fingerRight", "fist", "none", "palm", "thumb"]

#Fix the size of the images used by the neural network
img_size = 50

#Checking for each class name folder inside the dataset folder
for class_name in class_names:
   path = os.path.join(data_dir, class_name)
    #For each image in the class folder, read it and convert it into gray image
   for img in os.listdir(path):
       img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

trainImages = []

#Function to create an array of dataset
def createTrainingData():
    for class_name in class_names:
        path = os.path.join(data_dir, class_name)
        class_id = class_names.index(class_name)
        for img in os.listdir(path):
            try:
                #Convert into gray images
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                #Resize with the defined image size
                resize_array = cv2.resize(img_array, (img_size, img_size))
                #Append the image and the corresponding label into the array
                trainImages.append([resize_array, class_id])
            except Exception as e:
                pass

createTrainingData()

#Mix up the different images
random.shuffle(trainImages)

x_trains = []
y_labels = []

#Create an array for each images and one for each label
for feature, label in trainImages:
    x_trains.append(feature)
    y_labels.append(label)

#Convert into numpy array and reshape the image array
x_trains = np.array(x_trains).reshape(-1, img_size, img_size, 1)
y_labels = np.array(y_labels)

#Save the arrays as numpy model
pickle_out = open("../data/Datasets/trainModelImages", "wb")
pickle.dump(x_trains, pickle_out)
pickle_out.close()

pickle_out = open("../data/Datasets/trainModelLabels", "wb")
pickle.dump(y_labels, pickle_out)
pickle_out.close()

pickle_in = open("../data/Datasets/trainModelImages", "rb")
X = pickle.load(pickle_in)
