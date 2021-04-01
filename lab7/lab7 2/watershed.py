import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_location(i,row,col):
    if i == 0:
        (row,col) = (row-1,col-1)
    elif i == 1:
        (row,col) = (row-1,col)
    elif i == 2:
        (row,col) = (row-1,col+1)
    elif i == 3:
        (row,col) = (row,col-1)
    elif i == 4:
        pass
    elif i == 5:
        (row,col) = (row,col+1)
    elif i == 6:
        (row,col) = (row+1,col-1)
    elif i == 7:
        (row,col) = (row+1,col)
    elif i == 8:
        (row,col) = (row+1,col+1)
    return [row,col]



def getRegionalMinima(img):
    (height,width) = img.shape
    img = np.pad(img,((1,1),(1,1)),'constant',constant_values=(np.int32(255),np.int32(255)))
    marker = np.zeros((img.shape[0],img.shape[1]),dtype=np.int32)
    mark = 1
    for i in range(1,height+1):
        for j in range(1,width+1):
            neighbours = [img[i-1,j-1],img[i-1,j],img[i-1,j+1],img[i,j-1],img[i,j],img[i,j+1],img[i+1,j-1],img[i+1,j],img[i+1,j+1]]
            local_min = min(neighbours)
            if local_min == img[i,j]:
                [min_row,min_col] = i,j # If [i,j] is a local min, then it could be labeled in the marker
                marker[min_row,min_col] = mark
                mark += 1

    return marker[1:-1,1:-1]
                

def iterativeMinFollowing(img,markers):
    markers_copy = markers
    (height,width) = img.shape
    img = np.pad(img,((1,1),(1,1)),'constant',constant_values=(np.int32(255),np.int32(255)))
    markers_copy = np.pad(markers_copy,((1,1),(1,1)),'constant',constant_values=(np.int32(255),np.int32(255)))
    count = 1
    while (count != 0):
        count = 0
        for i in range (1,height+1):
            for j in range (1,width+1):
                if markers_copy[i,j] != 0 :
                    pass
                else:
                    neighbours = [img[i-1,j-1],img[i-1,j],img[i-1,j+1],img[i,j-1],img[i,j],img[i,j+1],img[i+1,j-1],img[i+1,j],img[i+1,j+1]]
                    local_min = min(neighbours)
                    [min_row,min_col] = get_location(neighbours.index(local_min),i,j)
                    if markers_copy[min_row,min_col] != 0:
                        markers_copy[i,j] = markers_copy[min_row,min_col]
                    else:
                        pass
        for i in markers_copy.flatten():
            if i == 0:
               count += 1
        print("The count is :"+str(count))
    
    final = markers_copy[1:-1,1:-1]
    return final


