import tensorflow as tf
import cv2

class_names = ["fingerLeft", "fingerRight", "fist", "none", "palm", "thumb"]

num_frames = 0
cam = cv2.VideoCapture(0)

model = tf.keras.models.load_model("../data/trainModel/CNN.model")

while True:
    ret, frame = cam.read()

    img_size = 50
    img_gray = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, (img_size, img_size))
    img_resize = img_resize/255.0
    img_reshape = img_resize.reshape(-1, img_size, img_size, 1)

    print(img_reshape)

    prediction = model.predict([img_reshape])
    prediction = list(prediction[0])
    class_name = class_names[prediction.index(max(prediction))]

    print(class_name)

    frame = cv2.flip(frame,1)
    cv2.putText(frame, class_name, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_4)
    cv2.imshow('Hand Gesture Recognition', frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()
cv2.destroyAllWindows()