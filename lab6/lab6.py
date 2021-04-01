# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:11:36 2018

@author: chenlin
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
def log_filter(blurred,sigma):
    k = 2 * round(3*sigma) + 1
    output = cv2.GaussianBlur(blurred,(k,k),sigma)
    output = cv2.Laplacian(output,cv2.CV_64F,ksize=k)
    return output

def main():
    #Read an image and convert it to grayscale
    img = cv2.imread("lab6.bmp")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   
    #Apply Gaussian filtering
    sigma = 2
    k = 2 * round(3*sigma) + 1
    blurred = cv2.GaussianBlur(img,(k,k),sigma)
    
    #Show both the original image and its blurred verion together in the same figure
    figure = plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(img)
    plt.title("input image")

    plt.subplot(2,1,2)
    plt.imshow(blurred)
    plt.title("blurred image")

    figure.tight_layout()    
    plt.show()
    #Create 3 level LoG volume by applying this filter at 3 different scales or sigma values to the blurred image obtained in step 2
    
    #Contruct 3 h*w*3 numpy array
    height = img.shape[0] # hieght of the input image
    width = img.shape[1] # width of the input image
    log_image = np.zeros((height,width,3),dtype=np.float32)

    sigma_values = [3,4,5]
    i = 1
    figure2 = plt.figure()
    for sigma in sigma_values:
        output = log_filter(blurred,sigma)
        plt.subplot(3,1,i)
        plt.imshow(output)
        plt.title("Level "+str(i))
        log_image[:,:,i-1] = output
        i = i + 1
    figure2.tight_layout()
    plt.show()

    #part 2
    sum_array = np.zeros((height,width,1))
    lm = scipy.ndimage.filters.minimum_filter(log_image,8)
    msk = (log_image == lm)
    #computing the sum of corresponding pixles in 3 channels    
    sum_array = np.sum(msk,axis=2)

    #Find all non-zeros (their location)
    location = np.nonzero(sum_array)
    #Plot red on all non-zero locations
    im = plt.imread("lab6.bmp")
    plt.imshow(im)
    
    x = location[1]
    y = location[0]
    plt.scatter(x,y,color='r',s=20)
    axes = plt.gca()
    axes.set_ylim([0,150])
    axes.set_xlim([0,300])
    plt.show()
    #Apply Otsu thresholding on the blurred image in step 2 of part 1
    ret,th = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #ret2 is the optimal threshold value

    #Remove all minima detected in part 2 where the pixel values in this image are less than this threshold
    x_array = np.array([])
    y_array = np.array([])

    for i in range (0,len(location[1])):
        if(img[location[0][i],location[1][i]] > ret):
            x_array = np.append(x_array,location[1][i])
            y_array = np.append(y_array,location[0][i])
    
    plt.imshow(im)
    plt.scatter(x_array,y_array,color='r',s=10)
    axes = plt.gca()
    axes.set_ylim([0,150])
    axes.set_xlim([0,300])
    plt.show()

    
                
                
            







main()
    
