import tensorflow as tf
import cv2


def class_action(name, recognized, counter):
    # Do an action depending on the pattern
    if recognized == name and recognized != "none":
        counter += 1

        if counter > 25 and recognized != "none":
            if recognized == "fingerLeft":
                print('fingerLeft action')
            if recognized == "fingerRight":
                print('fingerRight action')
            if recognized == "fist":
                print('fist action')
            if recognized == "palm":
                print('palm action')
            if recognized == "thumb":
                print('thumb action')

            recognized = "none"
            counter = 0
    else:
        recognized = name
        counter = 0
        counter += 1

    return recognized, counter

# Define the pattern classes
class_names = ["fingerLeft", "fingerRight", "fist", "none", "palm", "thumb"]

# Define class counters
class_recognized = 0
class_counter = 0

# Capture the webcam image
cam = cv2.VideoCapture(0)

# Load the neural network model
model = tf.keras.models.load_model("../data/trainModel/CNN.model")

# Region of interest (ROI) coordinates
top, right, bottom, left = 60, 420, 225, 590

while True:
    # Getting frame by frame
    ret, frame = cam.read()

    # Flip the frame
    frame = cv2.flip(frame, 1)

    # Get the ROI
    roi = frame[top:bottom, right:left]

    # Modify the frame to correspond of the input parameters used by the neural network
    img_size = 100
    img_gray = cv2.cvtColor(roi, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, (img_size, img_size))
    img_resize = img_resize / 255.0
    img_reshape = img_resize.reshape(-1, img_size, img_size, 1)

    # Predict the class_name for each frame
    prediction = model.predict([img_reshape])
    prediction = list(prediction[0])
    class_name = class_names[prediction.index(max(prediction))]
    # print(prediction)

    # Draw the ROI
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    # Write the classname on the frame and show it
    cv2.putText(frame, class_name, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_4)
    cv2.imshow('Hand Gesture Recognition', frame)

    class_recognized, class_counter = class_action(class_name, class_recognized, class_counter)

    # Define frame exit key
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()
cv2.destroyAllWindows()