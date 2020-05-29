import tensorflow as tf
import cv2

class_name = ["fingerLeft", "fingerRight", "fist", "none", "palm", "thumb"]

img_size = 50
img_array = cv2.imread("../data/dataTest/palm.jpg", cv2.IMREAD_GRAYSCALE)
array = cv2.resize(img_array, (img_size, img_size))
array = array/255.0
reshape_array = array.reshape(-1, img_size, img_size, 1)

print(reshape_array)

model = tf.keras.models.load_model("../data/trainModel/CNN.model")

prediction = model.predict([reshape_array])
prediction = list(prediction[0])
print(class_name[prediction.index(max(prediction))])

