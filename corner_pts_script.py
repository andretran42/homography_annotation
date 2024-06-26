import cv2
import numpy as np
import os
import pandas as pd

csv_file = "./data_labels.csv"

if __name__ == '__main__' :
    df = pd.read_csv(csv_file, header=None)
    im_dst = cv2.imread('ucla.png')
    # folder_path = "./img_data"

    # points_to_transform = np.array([[0, 221]], dtype=float)
    # transformed_points = cv2.perspectiveTransform(np.array([points_to_transform]), h)
    # print(transformed_points)
    for index, row in df.iterrows():
        im_path = "./img_done/" + row[0]
        im_src = cv2.imread(im_path)
        dim = im_src.shape
        height = dim[0]
        width = dim[1]
        h = np.array([
        [row[1],row[2],row[3]],  # Example values
        [row[4],row[5],row[6]],
        [row[7],row[8], 1.0]
        ], dtype=np.float32)
        p1 = np.array([[0, 0]], dtype=float) #top left
        p2 = np.array([[dim[1], 0]], dtype=float) #top right
        p3 = np.array([[dim[1], dim[0]]], dtype=float) #bottom right
        p4 = np.array([[0, dim[0]]], dtype=float) #bottom left
        wp1 = cv2.perspectiveTransform(np.array([p1]), h).reshape(2)
        wp2 = cv2.perspectiveTransform(np.array([p2]), h).reshape(2)
        wp3 = cv2.perspectiveTransform(np.array([p3]), h).reshape(2)
        wp4 = cv2.perspectiveTransform(np.array([p4]), h).reshape(2)
        df.loc[index, 10] = wp1[0]
        df.loc[index, 11] = wp1[1]
        df.loc[index, 12] = wp2[0]
        df.loc[index, 13] = wp2[1]
        df.loc[index, 14] = wp3[0]
        df.loc[index, 15] = wp3[1]
        df.loc[index, 16] = wp4[0]
        df.loc[index, 17] = wp4[1]
        # pts_src = np.array([p1, p2, p3, p4])
        # pts_dest = np.array([wp1, wp2, wp3, wp4])
        # print(pts_src)
        # print(pts_dest)
        # hm, status = cv2.findHomography(pts_src, pts_dest)
        # im_out = cv2.warpPerspective(im_src, hm, (im_dst.shape[1],im_dst.shape[0]))
    df.to_csv(csv_file, mode='w', header=False, index=False)
