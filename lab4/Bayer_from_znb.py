import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


# part I

img = cv2.imread('PeppersBayerGray.bmp', 0)

h,w = img.shape

# our final image will be a 3 dimentional image with 3 channels
rgb = np.zeros((h,w,3),np.uint8);


# reconstruction of the green channel IG

IG = np.copy(img) # copy the image into each channel

for row in range(0,h,4): # loop step is 4 since our mask size is 4.
    for col in range(0,w,4): # loop step is 4 since our mask size is 4.
        p_a = int(img[row,col])
        p_c = int(img[row,col+2])
        p_f = int(img[row+1,col+1])
        p_h = int(img[row+1,col+3])
        p_i = int(img[row+2,col])
        p_k = int(img[row+2,col+2])
        p_n = int(img[row+3,col+1])
        p_p = int(img[row+3,col+3])
        IG[row,col+1]=(p_a + p_c)/2
        IG[row,col+3]=(p_c + p_h)/2
        IG[row+1,col]=(p_a + p_i)/2
        IG[row+1,col+2]=(p_c+p_f+p_h+p_k)/4
        IG[row+2,col+1]=(p_f+p_i+p_k+p_n)/4
        IG[row+2,col+3]=(p_h+p_p)/2
        IG[row+3,col]=(p_i+p_n)/2
        IG[row+3,col+2]=(p_n+p_p)/2

# reconstruction of the red channel IR
IR = np.copy(img) # copy the image into each channel

for row in range(0,h,4): # loop step is 4 since our mask size is 4.
    for col in range(0,w,4): # loop step is 4 since our mask size is 4.

        p_b = int(img[row,col+1])
        p_d = int(img[row,col+3])
        p_j = int(img[row+2,col+1])
        p_l = int(img[row+2,col+3])
        IR[row,col+2]=(p_b + p_d)/2
        IR[row+1,col+1]=(p_b + p_j)/2
        IR[row+1,col+3]=(p_d+p_l)/2
        IR[row+1,col+2]=(p_b+p_d+p_j+p_l)/4
        IR[row+2,col+2]=(p_j+p_l)/2
        IR[row,col]=p_b
        IR[row+1,col]=IR[row+1,col+1]
        IR[row+2,col]=p_j
        IR[row+3,col]=p_j
        IR[row+3,col+1]=p_j
        IR[row+3,col+2]=IR[row+2,col+2]
        IR[row+3,col+3]=p_l
        

# reconstruction of the blue channel IB
IB = np.copy(img) # copy the image into each channel

for row in range(0,h,4): # loop step is 4 since our mask size is 4.
    for col in range(0,w,4): # loop step is 4 since our mask size is 4.

        p_e = int(img[row+1,col])
        p_g = int(img[row+1,col+2])
        p_m = int(img[row+3,col])
        p_o = int(img[row+3,col+2])
        IB[row+1,col+1]=(p_e + p_g)/2
        IB[row+2,col]=(p_e + p_m)/2
        IB[row+2,col+2]=(p_g+p_o)/2
        IB[row+2,col+1]=(p_e+p_g+p_m+p_o)/4
        IB[row+3,col+1]=(p_m+p_o)/2
        IB[row,col]=p_e
        IB[row,col+1]=IB[row+1,col+1]
        IB[row,col+2]=p_g
        IB[row,col+3]=p_g
        IB[row+1,col+3]=p_g
        IB[row+2,col+3]=IB[row+2,col+2]
        IB[row+3,col+3]=p_o


# merge the channels

rgb[:,:,0]=IR
rgb[:,:,1]=IG
rgb[:,:,2]=IB


cv2.imwrite('rgb.jpg',rgb);

plt.imshow(rgb),plt.title('rgb')
plt.show()


# part II should be written here:
IR = IR.astype(np.float32)
IG = IG.astype(np.float32)
IB = IB.astype(np.float32)
DR = IR - IG
DB = IB - IG
MR = cv2.medianBlur(DR,3)
MB = cv2.medianBlur(DB,3)
IRR = MR + IG
IBB = MB + IG
for i in range(IRR.shape[0]):
    for j in range(IRR.shape[1]):
        if (IRR[i][j] < 0):
            IRR[i][j] = 0
        elif (IRR[i][j] > 255):
            IRR[i][j] = 255

for m in range(IBB.shape[0]):
    for n in range(IBB.shape[1]):
        if (IBB[m][n] < 0):
            IBB[m][n] = 0
        elif (IBB[m][n] > 255):
            IBB[m][n] = 255
            
            
            
h,w = img.shape
            
# our final image will be a 3 dimentional image with 3 channels
argb = np.zeros((h,w,3),np.uint8)

argb[:,:,0]=IRR
argb[:,:,1]=IG
argb[:,:,2]=IBB


cv2.imwrite('argb.jpg',argb);

plt.imshow(argb),plt.title('argb')
plt.show()

