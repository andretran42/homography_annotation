import cv2
import numpy as np
import os
import pandas as pd
import random

folder_path = "./img_done"

if __name__ == '__main__' :
    # for filename in os.listdir(folder_path):
    filename = "field0368.png"
    im_path = os.path.join(folder_path, filename)
    im_src = cv2.imread(im_path)
    dim = im_src.shape
    height = dim[0]
    width = dim[1]
    dx1 = random.randint(-75, 0)
    dy1 = random.randint(-75, 0)
    dx2 = random.randint(0, 75)
    dy2 = random.randint(-75, 0)
    dx3 = random.randint(0, 75)
    dy3 = random.randint(0, 75)
    dx4 = random.randint(-75, 0)
    dy4 = random.randint(0, 75)
    print(height, width)
    print(dy4, dx4, dy3, dx3)
    p1 = np.array([[0, 0]], dtype=float) #top left
    p2 = np.array([[width, 0]], dtype=float) #top right
    p3 = np.array([[width, height]], dtype=float) #bottom right
    p4 = np.array([[0, height]], dtype=float) #bottom left
    wp1 = np.array([[p1[0][0]+dx1, p1[0][1]+dy1]])
    wp2 = np.array([[p2[0][0]+dx2, p2[0][1]+dy2]])
    wp3 = np.array([[p3[0][0]+dx3, p3[0][1]+dy3]])
    wp4 = np.array([[p4[0][0]+dx4, p4[0][1]+dy4]])
    pts_src = np.array([p1, p2, p3, p4])
    pts_dest = np.array([wp1, wp2, wp3, wp4])
    hm, status = cv2.findHomography(pts_src, pts_dest)
    im_out = cv2.warpPerspective(im_src, hm, (im_src.shape[1],im_src.shape[0]))

    cv2.imshow('Original Image', im_src)
    cv2.imshow('Warped Image', im_out)

# Wait for a key press and close the image windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

