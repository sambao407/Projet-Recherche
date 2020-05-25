import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# region of interest (ROI) coordinates
(top, right, bottom, left) = 10, 350, 225, 590

while 1:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    frame = cv2.flip(frame, 1)

    # get the ROI
    roi = frame[top:bottom, right:left]

    # draw the segmented hand
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # display the frame with segmented hand
    cv2.imshow("Camera Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
