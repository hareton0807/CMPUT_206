#Press q on keyboard to exit
import cv2
import numpy as np
global img
#Blur a localized area around the pixel where user has clicked
#parameters of mouseclick listener are give
def lBlur(event,x,y,flags,param):
       # DUBLE CLICK TO TRIGGER THIS EVENT
        if event == cv2.EVENT_LBUTTONDBLCLK:
                height = img.shape[0]
                width = img.shape[1]
                blurred = cv2.GaussianBlur(img,(35,35),0)
                # find out how to move the clcik point to the center of the mask
                right_move = width - x
                down_move = height - y
                # move all points in the way that the click point moves
                for i in range (height):
                    for j in range (width):
                        img[i][j] = np.add(np.multiply(Mask[i+down_move][j+right_move],blurred[i][j]),np.multiply((1-Mask[i+down_move][j+right_move]),img[i][j]))
                        

def computeCauchyMask(image):
        #The mask should be twice as large as the image

        #Get the height and width of the original image
        height = image.shape[0]
        width = image.shape[1]
        #Create an empty numpy array to store a blank mask /image
        Mask = np.zeros((2*height,2*width,1),np.float64)
        # Here we use numpy array to ensure that np.add and np.multiply will work later
        #Set standard deviation
        std = 50

        #Compute the weight in Gaussian distributionss
        # Weight is max at the center of THE MASK as the center of the mask is at
        # the same location as the clicked point in original image
        for x in range (Mask.shape[0]):
                for y in range (Mask.shape[1]):
                    Mask[x][y] = 1 / ( 1 + ( ( x - height )**2 + ( y - width )**2) / std**2)
                        
        return Mask




def main():

        global img,ix,iy,Mask
        
        #Open an grayscale image
        img = cv2.imread("ex1.jpg")


        #Name a window as image
        cv2.namedWindow("image")
        

        #Set mouseclick listener
        cv2.setMouseCallback("image",lBlur)
        Mask = computeCauchyMask(img)
     
        while (1):

                cv2.imshow("image",img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        cv2.destroyAllWindows()

main()
                
