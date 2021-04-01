 #Compute the cumulative histogram'
def cal_cumulative_histogram(i,my_array):
    i = int(i)
    if i==0:
        return 0
    elif i >= 256:
        return True
    else:
        return cal_cumulative_histogram(i-1,my_array) + my_array[i]
    
def main():
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    img = cv2.imread("test.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    my_array = np.array([0]*256)
    second_array = np.array([0]*256)
    
    (img_row,img_col) = img.shape
    for row in range (0,img_row):
        for col in range (0,img_col):
            pixel = img[row,col]
            my_array[pixel] += 1
            
    x = np.arange(0,255,1)
    y = my_array[x]
    plt.plot(x,y)
    plt.xlabel("intensity i")
    plt.ylabel("h(i)")
    plt.title("Histogram of test.jpg")
    plt.show()

    
    for row in range (0,img_row):
        for col in range (0,img_col):
            pixel = img[row,col]
            img[row,col] = math.floor((256-1)/(img_row*img_col)*cal_cumulative_histogram(pixel,my_array) + 0.5)
    

            
    cv2.imshow("image2",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    for row in range (0,img_row):
        for col in range (0,img_col):
            pixel = img[row,col]
            second_array[pixel] += 1
            
    x = np.arange(0,255,1)
    y = second_array[x]
    plt.plot(x,y)
    plt.xlabel("intensity i")
    plt.ylabel("h(i)")
    plt.title("Histogram of test2.jpg")
    plt.show()
                

    
    
   
    
    
main()
    