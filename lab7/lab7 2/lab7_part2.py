import numpy as np
import cv2
import matplotlib.pyplot as plt
import watershed

def imreconstruct(marker, mask):
    curr_marker = np.copy(marker).astype(mask.dtype)
    kernel = np.ones([3, 3])
    while True:
        next_marker = cv2.dilate(curr_marker, kernel, iterations=1)
        intersection = next_marker > mask
        next_marker[intersection] = mask[intersection]
        if np.array_equal(next_marker, curr_marker):
            return curr_marker
        curr_marker = np.copy(next_marker)
    return curr_marker


def imimposemin(marker, mask):
    # adapted from its namesake in MATLAB
    fm = np.copy(mask)
    fm[marker] = -np.inf
    fm[np.invert(marker)] = np.inf
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        range = float(np.max(mask) - np.min(mask))
        if range == 0:
            h = 0.1
        else:
            h = range * 0.001
    else:
        # Add 1 to integer images.
        h = 1
    fp1 = mask + h
    g = np.minimum(fp1, fm)
    return np.invert(imreconstruct(
        np.invert(fm.astype(np.uint8)), np.invert(g.astype(np.uint8))
    ).astype(np.uint8))

sigma = 2.5
img_name = 'lab7.bmp'
img_rgb = cv2.imread(img_name).astype(np.float32)
img_gs = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

img_blurred = cv2.GaussianBlur(img_gs, (int(2 * round(3 * sigma) + 1), int(2 * round(3 * sigma) + 1)), sigma,
                     borderType=cv2.BORDER_REPLICATE)

[img_grad_y, img_grad_x] = np.gradient(img_blurred)
img_grad = np.square(img_grad_x) + np.square(img_grad_y)

# refined blob locations generated generated in part 3 of lab 6
blob_markers = np.loadtxt('blob_markers.txt', dtype=np.bool, delimiter='\t')

img_grad_min_imposed = imimposemin(blob_markers, img_grad)

markers = watershed.getRegionalMinima(img_grad_min_imposed)
plt.figure(0)
plt.imshow(markers)
plt.title('markers')

labels = watershed.iterativeMinFollowing(img_grad_min_imposed, markers)
plt.figure(1)
plt.imshow(labels)
plt.title('labels')

plt.show()



#detect contours
img,contours,hierarchy = cv2.findContours(labels,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

#draw contours
##cv2.drawContours(img,contours,-1,(255,0,0),-1)
##plt.imshow(img)
##plt.title("contours")
##plt.show()

#prune the contours obtained so that only those whose areas are between 10 and 50 are retained
#check if it is an external contour and its area is b/w 10 and 50
new_contours = []
for i,c in enumerate(contours):
    if hierarchy[0][i][3] == -1 and cv2.contourArea(c) >= 10 and cv2.contourArea(c) <= 50:
        new_contours.append(c)
#draw contours
cv2.drawContours(img_gs,new_contours,-1,(0,255,0),-1)
plt.imshow(img_gs)
plt.title("pruned contours")
plt.show()
        
        





