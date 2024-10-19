import cv2 as cv
import numpy as np



##Hough Circle Detection Distance Function
dist = lambda x1, y1, x2, y2: (x1 - x2)**2 + (y1 - y2)**2 ##We will use the swuare of the distance to compare

## Opening the webcam on laptop
videoCapture = cv.VideoCapture(0, cv.CAP_DSHOW)
prevCircle = None

##Check if the camera was opened successfully
if videoCapture.isOpened() is False:
    print("Error: Camera was not opened successfully.")
    exit()

## Main Capture Loop
while True:
    ret, frame = videoCapture.read()

    if not ret: 
        print("Error: Frame was not captures succesfully.")
        break

    ##Applying grayscale and blur to the frame for better circle detection from HoughCirles
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (17, 17), 0)

    ##Using HoughCircles to detect circle in frame, set min distance at 100 due to wanting to only detect one circle
    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.1, 100, 
                              param1 = 100, param2 = 60, minRadius = 5, maxRadius = 400)
    
    #Condition that draws circle around the detected circle, and also chooses the best circle to track
    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) < dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                    chosen = i
        cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
        cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (255, 0, 255), 3)
        prevCircle = chosen

    cv.imshow("Circles", frame)

    if cv.waitKey(1) & 0xFF == ord('q'): 
        break

videoCapture.release()
cv.destroyAllWindows()
