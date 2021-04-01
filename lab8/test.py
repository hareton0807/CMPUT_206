import cv2
import numpy as np
from PIL import Image







import matplotlib.pyplot as plt
a = Image.open("A.jpg")
print(a.size)
##
##a = np.array(a)
##a[:,:,1] *= 0
##a[:,:,2] *= 0
##a = Image.fromarray(a)
##a.show()

red,green,blue = a.split()


matrix = []
matrix.append(list(red.getdata()))
matrix.append(list(green.getdata()))
matrix.append(list(blue.getdata()))

for ind in range(0,3):
    matrix[ind] = [matrix[ind][i:i+a.size[0]] for i in range (0,len(matrix[ind]),a.size[0])]

matrix = np.asarray(matrix)
print(np.shape(matrix))
print(matrix[0])
print(matrix[1])
print(matrix[2])
##red = matrix[0,:,:]
##blue = matrix[1,:,:]
##green = matrix[2,:,:]
##print(np.shape(red))


##im_final = np.zeros((512,512,3),dtype='int64')
##im_final[:,:,0] = blue
##im_final[:,:,1] = green
##im_final[:,:,2] = red
##plt.imshow(im_final)
##plt.show()

