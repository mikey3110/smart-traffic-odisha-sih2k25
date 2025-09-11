import cv2
import numpy as np
 
## Load pre-trained Haar Cascade classifiers for cars and buses.
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
bus_cascade = cv2.CascadeClassifier('haarcascade_bus.xml')
 
if car_cascade.empty() or bus_cascade.empty():
    print("Error: Could not load cascade classifier XML file.")
    exit()
 
cap = cv2.VideoCapture('vehicles.mp4')
 
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
 
while True:
    ## Read a frame from the video stream.
    ret, frame = cap.read()
    if not ret:
        break
 
    ## Convert the frame to grayscale for faster processing.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    ## Detect cars in the frame.
    ## The 'detectMultiScale' function returns a list of rectangles (x, y, w, h).
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
 
    ## Detect buses in the frame.
    buses = bus_cascade.detectMultiScale(gray, 1.1, 1)
 
    ## Draw rectangles around the detected cars.
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Car', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
    ## Draw rectangles around the detected buses.
    for (x, y, w, h) in buses:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Bus', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
 
    ## Display the resulting frame.
    cv2.imshow('Vehicle Detector', frame)
 
    ## Break the loop if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
## Release the video capture object and close all windows.
cap.release()
cv2.destroyAllWindows()
