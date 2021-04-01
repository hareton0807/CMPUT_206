# Press q on keyboard to exit
import cv2
import numpy as np

def nothing(x):
    pass

img = cv2.imread("ex1.jpg")
imgCopy = cv2.imread("ex1.jpg")
cv2.namedWindow("image")

#Create trackbars for threshold value changes
cv2.createTrackbar("MaxVal","image",0,300,nothing)
cv2.createTrackbar("MinVal","image",0,300,nothing)


while (1):
    cv2.imshow("image",img)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

    #Get current positions of the above trackbars
    maxVal = cv2.getTrackbarPos("MaxVal","image")
    minVal = cv2.getTrackbarPos("MinVal","image")

    img = cv2.Canny(imgCopy,minVal,maxVal)
cv2.destroyAllWindows() 
    

