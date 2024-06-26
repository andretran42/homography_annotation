import cv2
import numpy as np
import os
import pandas as pd
import random

folder_path = "./img_done"
folder_path2 = "./img_done2"
df = pd.read_csv("./data_labels.csv", header=None)
csv_file = "./data_labels.csv"
if __name__ == '__main__' :
    j = 1
    new_df = pd.read_csv("./data_labels2.csv", header=None)
    for index, row in new_df.iterrows():
        for i in range(1,10):
            im_path = "./img_done/" + row[0]
            print(im_path)
            im_src = cv2.imread(im_path)
            dim = im_src.shape
            height = dim[0]
            width = dim[1]
            dx1 = random.randint(-98, 10)
            dy1 = random.randint(-98, 10)
            dx2 = random.randint(-10, 98)
            dy2 = random.randint(-98, 10)
            dx3 = random.randint(-10, 98)
            dy3 = random.randint(-10, 98)
            dx4 = random.randint(-98, 10)
            dy4 = random.randint(-10, 98)
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
            new_name = "field" + '{:04d}'.format(i*663+j) + ".png"
            file_path = os.path.join(folder_path2, new_name)
            cv2.imwrite(file_path, im_out)
            wwp1 = cv2.perspectiveTransform(np.array([wp1]), hm).reshape(2)
            wwp2 = cv2.perspectiveTransform(np.array([wp2]), hm).reshape(2)
            wwp3 = cv2.perspectiveTransform(np.array([wp3]), hm).reshape(2)
            wwp4 = cv2.perspectiveTransform(np.array([wp4]), hm).reshape(2)

            owp1 = np.array([row[10], row[11]])
            owp2 = np.array([row[12], row[13]])
            owp3 = np.array([row[14], row[15]])
            owp4 = np.array([row[16], row[17]])

            pts_dest2 = np.array([owp1, owp2, owp3, owp4])
            pts_src2 = np.array([wwp1, wwp2, wwp3, wwp4])
            new_hm, status2 = cv2.findHomography(pts_src2, pts_dest2)
            newwp1 = cv2.perspectiveTransform(np.array([p1]), new_hm).reshape(2)
            newwp2 = cv2.perspectiveTransform(np.array([p2]), new_hm).reshape(2)
            newwp3 = cv2.perspectiveTransform(np.array([p3]), new_hm).reshape(2)
            newwp4 = cv2.perspectiveTransform(np.array([p4]), new_hm).reshape(2)
            df.loc[new_name, 10] = newwp1[0]
            df.loc[new_name, 11] = newwp1[1]
            df.loc[new_name, 12] = newwp2[0]
            df.loc[new_name, 13] = newwp2[1]
            df.loc[new_name, 14] = newwp3[0]
            df.loc[new_name, 15] = newwp3[1]
            df.loc[new_name, 16] = newwp4[0]
            df.loc[new_name, 17] = newwp4[1]
            df.loc[new_name, 0] = new_name
            print(new_name)
        j += 1
    df.to_csv(csv_file, mode='w', header=False, index=False)

