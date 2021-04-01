import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('noisy.jpg')


kernel = np.ones((3,3),np.float32)/9
dst = cv2.filter2D(img,-1,kernel)
''''img_row = img.shape[0]
img_col = img.shape[1]
for row in range(1,img_row):
    for col in range(1,img_col):
       dst[row,col] =  img[row,col] - dst[row,col
       '''
       
dst = img + (img-dst)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Filtered3')
plt.xticks([]), plt.yticks([])
plt.show()