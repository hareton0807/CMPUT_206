def main():
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    img = cv2.imread("noisy.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel = np.zeros((3,3),int)
    kernel[1][1] = 1
   
    (img_row,img_col) = img.shape


    blank_image = np.zeros((img_row,img_col,1))
    
    k = 1
    for i in range(k,img_row-2*k):
        for j in range(k,img_row-2*k):
                for l in range(-k,k):
                    for m in range(-k,k):
                        blank_image[i,j] = blank_image[i,j] + img[i+l,j+m] * kernel[l+k,m+k]

    cv2.imshow("image",blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


                        

main()
    
    
