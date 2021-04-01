import cv2
import numpy as np
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import math

    

def main():
    
    
    # Read an grayscale image
    img = cv2.imread("frame180.jpg",0)
##    cv2.imshow("Gradient magnitude image",mag)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()

    

##    # Display the image
##    cv2.imshow("image",img)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()

    #horizontal edges
    kernel = np.zeros((3,3),int)
    
    kernel[0][0] = 1
    kernel[1][0] = 2
    kernel[2][0] = 1
    kernel[0][2] = -1
    kernel[1][2] = -2
    kernel[2][2] = -1
    #print(kernel)
    dst = cv2.filter2D(img,-1,kernel)

    #vertical edges
    kernel2 = np.zeros((3,3),int)
    kernel2[0][0] = 1
    kernel2[0][1] = 2
    kernel2[0][2] = 1
    kernel2[2][0] = -1
    kernel2[2][1] = -2
    kernel2[2][2] = -1
    dst2 = cv2.filter2D(img,-1,kernel2)

    # depth
    kernel3 = np.zeros((3,3,3),int)
    f0 = kernel3[0]
    f0[0][0] = f0[0][2] = f0[2][0] = f0[2][2] = 1
    f0[0][1] = f0[1][0] = f0[1][2] = f0[2][1] = 2
    f0[1][1] = 4

    f2 = kernel3[2]
    f2[0][0] = f2[0][2] = f2[2][0] = f2[2][2] = -1
    f2[0][1] = f2[1][0] = f2[1][2] = f2[2][1] = -2
    f2[1][1] = -4

    print("k3: ",kernel3)

    # c processing


##    dx = ndimage.sobel(img, 0)  # x derivative
##    dy = ndimage.sobel(img, 1)  # y derivative
##    #dz = ndimage.sobel(your3Dmatrix, 2)  # z derivative
##    result = abs(dx) + abs(dy)
##
##    result = np.array(result,dtype=np.uint8)
##    cv2.imshow("gradient",result)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
##    
##
##    r2 = ndimage.sobel(img)
##    r2 = np.array(r2,dtype=np.uint8)
##    cv2.imshow("gradient",r2)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
##
##    compare = result - r2
##    print(compare[0:10,0:10])


    
    #Display those two images
    cv2.imshow("vertical edges",dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("horizontal edges",dst2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Compute magnitude
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = abs(dst[i][j]) + abs(dst2[i][j])


    #Display the gradient magnitude image
    cv2.imshow("Gradient magnitude image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
    


main()
    
    
    
    
    
        
