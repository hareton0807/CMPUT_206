def main():
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    img_day = cv2.imread("day.jpg")
    img_day = cv2.cvtColor(img_day,cv2.COLOR_BGR2GRAY)
    img_night = cv2.imread("night.jpg")
    img_night = cv2.cvtColor(img_night,cv2.COLOR_BGR2GRAY)
    
#    plt.hist(img_day.ravel(),256,[0,256])
#    plt.title("Histogram of the day image")
#    plt.show()
#    plt.hist(img_night.ravel(),256,[0,256])
#    plt.title("Histogram of the night image")
#    plt.show()    
    
    my_array = np.array([0]*256)
    (day_row,day_col) = img_day.shape
    for row in range (0,day_row):
        for col in range (0,day_col):
            pixel = img_day[row,col]
            my_array[pixel] += 1
    my_array = my_array / (day_row*day_col) 
    
    
    second_array = np.array([0]*256)
    (night_row,night_col) = img_night.shape
    for row in range (0,night_row):
        for col in range (0,night_col):
            pixel = img_night[row,col]
            second_array[pixel] += 1
    second_array = second_array / (night_row*night_col) 
    
    BC = 0    
    for i in range (0,255):
        BC += math.sqrt(my_array[i]*second_array[i])     
    
    print(BC)
    
    
main()