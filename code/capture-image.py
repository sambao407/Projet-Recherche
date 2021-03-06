import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("Camera")

#Region of interest (ROI) coordinates
top, right, bottom, left = 60, 420, 225, 590

img_counter = 0

while True:
    ret, frame = cam.read()

    frame = cv2.flip(frame, 1)
    # Draw the ROI
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Camera", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "../data/p/hand_gesture_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} created!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()