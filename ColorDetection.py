import cv2 as cv
import numpy as np
from ultralytics import YOLO

#Physics constants
g = 9.81
fps = 30
dt = 1/fps

#tracking variables
prev_position = None
vx, vy = 0, 0

#Load Yolo Model 11
model = YOLO("yolo11n.pt")

def draw_trajectory(frame, x0, y0, vx, vy, time_steps=60):
    points = []
    #Calculate the trajectory points
    for t in range(time_steps):
        x = int(x0 + vx*t)
        y = int(y0 + vy*t + 0.5*g*t**2)
        points.append((x, y))
    #Draw the trajectory
    for i in range(1, len(points)):
        cv.line(frame, points[i-1], points[i], (0, 255, 0), 2)

## Opening the webcam on laptop
videoCapture = cv.VideoCapture(0, cv.CAP_DSHOW)

## Check if the camera was opened successfully
if videoCapture.isOpened() is False:
    print("Error: Camera was not opened successfully.")
    exit()

## Main Capture Loop
while True:
    ret, frame = videoCapture.read()

    if not ret: 
        print("Error: Frame was not captured successfully.")
        break
    
    #load results from YOLO model
    results = model(frame, stream=True)

    #extract and draw boundary boxes
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    ## Convert frame to HSV color space
    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    ## Define range for white color in HSV space
    lower_white = np.array([0, 0, 180], dtype=np.uint8)  # Minimum value for white
    upper_white = np.array([180, 60, 255], dtype=np.uint8)  # Maximum value for white

    ## Threshold the HSV image to get only white colors
    mask = cv.inRange(hsvFrame, lower_white, upper_white)

    ## Find contours of the white object
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    ## If any contours are found, draw a rectangle around the largest white area
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        (x,y), radius = cv.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        cv.circle(frame, center, int(radius), (255, 0, 0), 2)

        #Calculating velocity of previous position
        # vx = (currentPos-prevPos)/dt
        #vy = (currentPos-prevPos)/dt
        if prev_position is not None:
            vx = (center[0] - prev_position[0]) / dt
            vy = (center[1] - prev_position[1]) / dt
        

        #Predict next position
        #t: Prediction position will be placed at time elapsed t seconds
        #x_pred = x0 + u*t
        #y_pred = y0 + u*t + 0.5*a*t^2
        t = 1.0
        x_pred = int(center[0] + vx*t)
        y_pred = int(center[1] + vy*t + 0.5*g*t**2)
        print(x_pred)
        print(y_pred)

        #Draw circle using predicted values
        cv.circle(frame, (x_pred, y_pred), 10, (0, 255, 0), 3)

        #Draw prediction arc
        draw_trajectory(frame, center[0], center[1], vx, vy)

        #Update previous position
        prev_position = center

    else:
        #for no ball detcted do not make prediction
        prev_position = None


    ## Show the frame with detected white areas
    cv.imshow("Projectile Prediciton", frame)

    ## Exit on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break

videoCapture.release()
cv.destroyAllWindows()
