import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('noisy.jpg')

kernel = -np.zeros((3,3),np.float32)
kernel[1][1] = 1
dst = cv2.filter2D(img,-1,kernel)


plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Filtered_1')
plt.xticks([]), plt.yticks([])
plt.show()
plt.close('all')

kernel_2 = -np.zeros((3,3),np.float32)
kernel_2[1][2] = 1
dst_2 = cv2.filter2D(img,-1,kernel_2)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Filtered_2')
plt.xticks([]), plt.yticks([])
plt.show()
plt.close('all')

