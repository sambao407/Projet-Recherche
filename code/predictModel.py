import tensorflow as tf
import cv2
# from PIL import crop

class_names = ["fingerLeft", "fingerRight", "fist", "none", "palm", "thumb"]

#Capture the webcam image
cam = cv2.VideoCapture(0)

#Load the neural network model
model = tf.keras.models.load_model("../data/trainModel/CNN.model")

#Region of interest (ROI) coordinates
top, right, bottom, left = 60, 420, 225, 590

start_recording = False

while True:
    #Getting frame by frame
    ret, frame = cam.read()

    #Modify the frame to correspond of the input parameters used by the neural network
    img_size = 50
    img_gray = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, (img_size, img_size))
    img_resize = img_resize/255.0
    img_reshape = img_resize.reshape(-1, img_size, img_size, 1)

    #Predict the class_name for each frame
    prediction = model.predict([img_reshape])
    prediction = list(prediction[0])
    class_name = class_names[prediction.index(max(prediction))]
    print(prediction)

    #Flip the frame
    frame = cv2.flip(frame,1)

    #Get the ROI
    roi = frame[top:bottom, right:left]
    #Draw the ROI
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    #Write the classname on the frame and show it
    cv2.putText(frame, class_name, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_4)
    cv2.imshow('Hand Gesture Recognition', frame)
    cv2.imshow('Class Detection', roi)

    #Define frame exit key
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()
cv2.destroyAllWindows()