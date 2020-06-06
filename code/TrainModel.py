import cv2
import tensorflow as tf
import pickle
from tensorflow.keras import layers, models

# Open the numpy model files
trainImages = pickle.load(open("../data/Datasets/trainModelImages", "rb"))
trainLabels = pickle.load(open("../data/Datasets/trainModelLabels", "rb"))

print(trainImages.shape)
# Normalizing the data (a pixel goes from 0 to 255)
trainImages = trainImages/255.0

# Start to build the neural network
model = models.Sequential()

# Adding 3 2D convolutional layers
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

# Adding 2 hidden layers
model.add(layers.Flatten())
model.add(layers.Dense(128))
model.add(layers.Activation("relu"))

model.add(layers.Dense(128))
model.add(layers.Activation("relu"))

# Adding the output layer with 6 neurones, for the 6 classes
model.add(layers.Dense(6))
model.add(layers.Activation("softmax"))

# Display the summarry of the neural network configuration
model.summary()

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Training the model
# You can change the number of iteration with the epoch parameter
# Validation_split corresponds to the percentage of images used for the validation phase compare
history = model.fit(trainImages, trainLabels, batch_size=32, epochs=50, validation_split=0.1)

#Save the neural network model
modelTrain = model.to_json()
with open("../data/trainModel/model.json", "w") as json_file:
    json_file.write(modelTrain)
model.save_weights("../data/trainModel/model.h5")
print("model saved")
model.save("../data/trainModel/CNN.model")

print(history.history.keys())