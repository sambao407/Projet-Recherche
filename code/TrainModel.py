import cv2
import tensorflow as tf
import pickle
from tensorflow.keras import layers, models

trainImages = pickle.load(open("../data/Datasets/trainModelImages", "rb"))
trainLabels = pickle.load(open("../data/Datasets/trainModelLabels", "rb"))

print(trainImages.shape)

trainImages = trainImages/255.0

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=trainImages.shape[1:]))
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(128))
model.add(layers.Activation("relu"))

model.add(layers.Dense(128))
model.add(layers.Activation("relu"))

model.add(layers.Dense(6))
model.add(layers.Activation("softmax"))

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(trainImages, trainLabels, batch_size=32, epochs=50, validation_split=0.1)

modelTrain = model.to_json()
with open("../data/trainModel/model.json", "w") as json_file:
    json_file.write(modelTrain)

model.save_weights("../data/trainModel/model.h5")
print("model saved")
model.save("../data/trainModel/CNN.model")

print(history.history.keys())

