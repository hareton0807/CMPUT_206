# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:50:36 2018

@author: ztang4
"""

def main():
    
    
    import cv2
    
    damaged_img = cv2.imread("damaged_cameraman.bmp")
    mask = cv2.imread("damage_mask.bmp")
    
   
    
    #Gaussian Filter has to run 25 times
    for i in range(0,50):
        img2 = cv2.GaussianBlur(damaged_img,(3,3),0)
    
    #Locate the coordinates of the damaged pixels and put them into the list
        mask_row = mask.shape[0]
        mask_col = mask.shape[1]
        
        damaged_img_row = damaged_img.shape[0]
        damaged_img_col = damaged_img.shape[1]
        for row in range(1,mask_row):
            for col in range(1,mask_col):
                if mask[row,col][0] == 0:
                    if mask[row,col][1] == 0:
                        if mask[row][col][2] == 0:                #Replace the damaged pixels with the new blurred pixels
                            damaged_img[row,col] = img2[row,col]
    cv2.imshow("image",damaged_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
main()
    
    
    
