import cv2 as cv
import numpy as np

# Open the default camera
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the resulting frame
    cv.imshow('Camera', frame)

    # Wait for 'q' key to exit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()