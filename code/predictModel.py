import tensorflow as tf
import cv2
import numpy as np

class_names = ["fingerLeft", "fingerRight", "fist", "none", "palm", "thumb"]

to_predict = []
num_frames = 0
cam = cv2.VideoCapture(0)

model = tf.keras.models.load_model("../data/trainModel/CNN.model")

while True:
    ret, frame = cam.read()

    img_size = 50
    img_gray = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, (img_size, img_size))
    img_resize = img_resize/255.0
    to_predict.append(img_resize)


    if len(to_predict) == 30:
        frame_to_predict = np.array(to_predict, dtype=np.uint8)
        frame_to_predict = frame_to_predict.reshape(-1, img_size, img_size, 1)
        prediction = model.predict([frame_to_predict])
        prediction = list(prediction[0])
        class_name = class_names[prediction.index(max(prediction))]

        print(class_name)

        cv2.putText(frame, class_name, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyWindow()


