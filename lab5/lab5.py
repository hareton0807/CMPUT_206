import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb

def flatten(input):
  new_list = []
  for i in input:
    for j in i:
        new_list.append(j)
  return new_list       
    

MIN_MATCH_COUNT = 10

img1 = cv2.imread('im1.jpg', 0)
img2 = cv2.imread('im2.jpg',0)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]


# Initiate BRISK detector
brisk = cv2.BRISK_create()

# Find the keypoints and descriptors with BRISK
(kpt,desc) = brisk.detectAndCompute(img1,None)
bk_img = img1.copy()
out_img = img1.copy()
out_img = cv2.drawKeypoints(bk_img,kpt,out_img)
cv2.imshow("out_image",out_img)
while (1):
  if cv2.waitKey(10) & 0xFF == ord("q"):
    break
cv2.destroyAllWindows()

(kpt2,desc2) = brisk.detectAndCompute(img2,None)
bk_img2 = img2.copy()
out_img2 = img2.copy()
out_img2 = cv2.drawKeypoints(bk_img2,kpt2,out_img2)
cv2.imshow("out_image2",out_img2)
while (1):
  if cv2.waitKey(10) & 0xFF == ord("q"):
    break
cv2.destroyAllWindows()

# initialize Brute-Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)

# use KNN match of Brute-Force matcher for descriptors
matches = bf.knnMatch(desc,desc2,k=2)
#Chenlin: Apply ratio test to exclude outliers

good = []
for m,n in matches:
  if m.distance < 0.75*n.distance:
    good.append(m)
  

# Compute homography matrix M
if len(good) > MIN_MATCH_COUNT:
  dst_pts = np.float32([ kpt[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
  src_pts = np.float32([ kpt2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
  M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
  matchesMask = mask.ravel().tolist()

  pts = np.float32([ [0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0] ]).reshape(-1,1,2)
  dst = cv2.perspectiveTransform(pts,M)

  img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
  print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
  matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

#img3 = cv2.drawMatches(img1,kpt,img2,kpt2,good,None,**draw_params)

#plt.imshow(img3, 'gray'),plt.show()    


#if match is done (# your code #):
 
  # Initialize a matrix to include all the coordinates in the image, from (0, 0), (1, 0), ..., to (w-1, h-1)
  # In this way, you do not need loops to access every pixel
  
  # Calculate the new image coordinates based on the homography matrix
c = np.zeros((3, h2*w2), dtype=np.int)
for y in range(h2):
   c[:, y*w2:(y+1)*w2] = np.matrix([np.arange(w2), [y] * w2,  [1] * w2])
new_c = M * np.matrix(c)
new_c = np.around(np.divide(new_c, new_c[2]))

  # The new coordinates may have negative values. So perform translation if necessary
x_min = int(np.amin(new_c[0]))
y_min = int(np.amin(new_c[1]))
x_max = int(np.amax(new_c[0]))
y_max = int(np.amax(new_c[1]))
if x_min < 0:
  t_x = -x_min
else:
  t_x = 0
if y_min < 0:
  t_y = -y_min
else:
  t_y = 0

  # Initialize the final image to include every pixel of the stitched images  
new_w = int(np.maximum(x_max, w1) - np.minimum(x_min, 0) + 1)
new_h = int(np.maximum(y_max, h1) - np.minimum(y_min, 0) + 1)

new_img1 = np.zeros((new_h, new_w), dtype=np.uint8)
new_img2 = np.zeros((new_h, new_w), dtype=np.uint8)

  # Assign the first image
new_img1[t_y:t_y+h1, t_x:t_x+w1] = img1

  # Assign the second image based on the newly calculated coordinates
for idx in range(c.shape[1]):
  x = c[0, idx]
  y = c[1, idx]
  x_c = int(new_c[0, idx])
  y_c = int(new_c[1, idx])
  new_img2[y_c + t_y, x_c + t_x] = img2[y, x]

  # The stitched image
new_img = (new_img1 + new_img2) / 2
cv2.imwrite('stitched_img.jpg', new_img);
cv2.imshow("Stitched Image", new_img)
cv2.waitKey()
cv2.destroyAllWindows()
