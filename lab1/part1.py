def main():
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    
    img = cv2.imread("test.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
#    cv2.imshow("image",img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    my_array = np.array([0]*256)
    
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
    plt.title("Histogram of test.jpg by my own code")
    plt.show()
    

    
    plt.hist(img.ravel(),256,[0,256])
    plt.xlabel("intensity i")
    plt.ylabel("h(i)")
    plt.title("Histogram of test.jpg by matplotlib")
    plt.show()
    
main()
