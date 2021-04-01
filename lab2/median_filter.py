def main():
    import numpy as np
    import cv2
    while (1):
        img = cv2.imread("panorama.jpeg")
        img = cv2.medianBlur(img,3)
        cv2.imwrite("new_image.jpeg",img)
        break
##        cv2.namedWindow("image")
##        cv2.resizeWindow("image",400,400)
##        cv2.imshow("image",img)
##        if cv2.waitKey(20) & 0xff == ord("q"):
##            break
##        cv2.destroyAllWindows()
        


##    img = cv2.imread("panorama.jpeg")
##    img = cv2.GaussianBlur(img,(3,3),0)
##    while (1):
##        cv2.imshow("image",img)
##        if cv2.waitKey(20) & 0xFF == 27:
##            break
##    cv2.destroyAllWindows()



    
	
	
main()



