from scipy import ndimage
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
img = mpimg.imread("ex1.jpg")
print(img.shape)


# calculate x derivative
##dx = ndimage.sobel(img,axis=0,mode='constant')
##plt.imshow(dx,cmap='gray')
##plt.savefig("x")
##plt.show()
##
##dy = ndimage.sobel(img,axis=1,mode='constant')
###mag = np.sqrt(dx**2+dy**2)

# disp mag
#plt.imshow(mag)
#plt.show()

##result = ndimage.sobel(img)


#result = np.hypot(dx,dy)


##plt.imshow(result,cmap='gray')
##plt.savefig("ndimage")
##plt.show()

mag = ndimage.generic_gradient_magnitude(img, ndimage.sobel)
print(mag[0:10,0:10])
##plt.imshow(mag)
##plt.savefig("mag",cmap='gray')
##plt.show()

# use np.stack to obtain a stack of 4 images



