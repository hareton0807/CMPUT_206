# Jpeg decoding

import cv2
import numpy as np
import math
# import zigzag functions
from zigzag import *

##################### step 5 #####################
# load h, w, block_size and padded_img from txt files

# load 'encoded.txt' into padded_img matrix.
# You should use np.loadtxt if you have already used np.savetxt to save them.
padded_img = np.loadtxt('encoded.txt',delimiter=',')##### your code #####


# load 'size.txt' to get [h, w, block_size]
# You should use np.loadtxt if you have already used np.savetxt to save them.
[h, w, block_size] = np.loadtxt('size.txt',delimiter=',')##### your code #####

##################### step 6 #####################
# get the size of padded_img
[H, W] = np.shape(padded_img)##### your code #####

# compute number of blocks by diving height and width of image by block size
# copy from step 1
# number of blocks in height
block_size = 8
nbh = H / block_size##### your code #####

# number of blocks in width
nbw = W / block_size##### your code #####

##################### step 7 #####################
# start decoding:
# divide encoded image into block size by block size (here: 8-by-8) blocks
# reshape it to one dimensional array (here: 64)
# use inverse zig-zag to reorder the array into a block
# apply 2D inverse discrete cosine transform


# iterate over blocks
for i in range(int(nbh)):
    
        # Compute start row index of the block, same as encoder
        row_ind_1 = i * block_size##### your code #####
        
        # Compute end row index of the block, same as encoder
        row_ind_2 = (i+1) * block_size##### your code #####
        
        for j in range(int(nbw)):
            
            # Compute start column index of the block, same as encoder
            col_ind_1 = j*block_size ##### your code #####
            
            # Compute end column index of the block, same as encoder
            col_ind_2 = (j+1) * block_size##### your code #####
            
            # select the current block we want to process using calculated indices
            block = padded_img[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]
            
            # reshape the 2D block (here: 8-by-8) to one dimensional array (here: 64)
            reshaped= block.reshape(64,)##### your code #####
            
            # use inverse_zigzag function to scan and reorder the array into a block
            reordered = inverse_zigzag(reshaped,8,8)##### your code #####
            
            # apply 2D inverse discrete cosine transform to the reordered matrix
            IDCT = cv2.dct(reordered, flags = cv2.DCT_INVERSE)##### your code #####
            
            # copy IDCT matrix into padded_img on current block corresponding indices
            padded_img[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = IDCT.reshape(8,8)##### your code #####


padded_img = np.uint8(padded_img)
cv2.imshow('decoded padded image', padded_img)

##################### step 8 #####################
# get the original size image (h by w) from padded_img

decoded_img = padded_img[0:int(h),0:int(w)]##### your code #####

cv2.imshow('decoded image', decoded_img)

##################################################

cv2.waitKey(0)
cv2.destroyAllWindows()




