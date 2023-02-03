import cv2 
import numpy as np 
cap = cv2.VideoCapture(0) 
while True:
    ret, img = cap.read() 
    kernel = np.ones([10, 10])
    kernel[:5, :] = -1 
    kernel[5:, :] = 1
    img = cv2.filter2D(img, -1, kernel)
    cv2.imshow("win", img) 
    cv2.waitKey(100)