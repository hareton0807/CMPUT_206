# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 17:49:31 2018

@author: chenlin
"""

import cv2
import numpy as np

img = cv2.imread("coins.png")
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('original',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

circles = cv2.HoughCircles(cimg,method = cv2.HOUGH_GRADIENT,dp = 1,minDist = 20,param1 = 80,param2=60,minRadius = 20 )
circles = np.uint16(np.around(circles))


for i in circles[0,:]:
    #draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    #draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

cv2.imwrite('final_part2.jpg',img)   
cv2.imshow('detected circles',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
