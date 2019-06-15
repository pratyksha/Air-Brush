import cv2 #importing the necessary libraries
import numpy as np
from collections import deque 

lwr = np.array([56, 51, 0]) # lower hsv value of the green pointer
uppr = np.array([75, 255, 255]) # upper hsv value of the green pointer

kernel = np.ones((5, 5), np.uint8)

brushQ = [deque(maxlen=512)]
brush_index = 0

window = np.zeros((471,636,3)) + 255

# Draw buttons on the white image
window = cv2.rectangle(window, (40,1), (140,65), (0,0,0), 2)
window = cv2.rectangle(window, (505,1), (600,65), (0,0,255), -1)

# Label the rectanglular boxes drawn on the image
cv2.putText(window, "Erase all", (50, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(window, "Exit", (530, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

cap = cv2.VideoCapture(0) # Video capture from webcam

while True:
    _, frame = cap.read() # Reading the the captured image as frame
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Draw buttons on the image
    frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
    frame = cv2.rectangle(frame, (505,1), (600,65), (0,0,255), -1)
    # Label the rectanglular boxes drawn on the image
    cv2.putText(frame, "Erase all", (50, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Exit", (530, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    
    mask = cv2.inRange(hsv, lwr, uppr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    ret,thresh = cv2.threshold(mask,100,150,0)

    _,contours,_ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]
    else:
        contour = np.array(contours)
    (x,y), radius = cv2.minEnclosingCircle(contour)
    cv2.circle(frame, (int(x), int(y)), int(radius), (250, 255, 0), 2)
    moment = cv2.moments(contour)
    center = (int(moment['m10'] / moment['m00']), int(moment['m01'] / moment['m00']))

    if center[1] > 65:
        brushQ[brush_index].appendleft(center)
        
    else:
        if 40 <= center[0] <= 140: # Erase all
            brushQ = [deque(maxlen=512)]
            brush_index = 0
            window[67:,:,:] = 255
        elif 505 <= center[0] <= 600:
            break#brushQ[brush_index].appendleft(center)
        
    for i in range(len(brushQ)):
        for j in range(1, len(brushQ[i])):
            if brushQ[i][j-1] is None or brushQ[i][j] is None:
                continue
            cv2.line(window, brushQ[i][j-1], brushQ[i][j], (0, 0, 0), 2)
            cv2.line(frame, brushQ[i][j-1], brushQ[i][j], (0, 0, 0), 2)
            
    cv2.imshow('Airbrush', window)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord(' '): # Pressing space_bar closes the tab
        break
  
cap.release()
cv2.destroyAllWindows()

    
    
