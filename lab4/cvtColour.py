import cv2
import matplotlib.pyplot as plt

img = cv2.imread("PeppersBayerGray.bmp",0)
cv2.imshow("original image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
dst_img = cv2.cvtColor(img,cv2.COLOR_BAYER_GR2RGB)
cv2.imshow("destination image",dst_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


