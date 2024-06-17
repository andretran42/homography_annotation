import cv2
import numpy as np
import os
import pandas as pd

rainbow_colors = [
    (238, 130, 238),  # Violet
    (75, 0, 130),     # Indigo
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 255, 255),    # Yellow
    (0, 165, 255),    # Orange
    (0, 0, 255)       # Red
]
color_index = 0
color_index2 = 0

csv_file = "./data_labels.csv"

 
if __name__ == '__main__' :
    im_src = cv2.imread('test_field.png')
    im_dst = cv2.imread('ucla.png')
    points_src = np.empty((0, 2), int)
    points_dest = np.empty((0, 2), int)
    def get_coordinates(event, x, y, flags, param):
        global color_index, points_src
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            cv2.circle(im_src, (x, y), 6, rainbow_colors[color_index], -1)
            color_index = (color_index + 1) % len(rainbow_colors)
            points_src = np.append(points_src, np.array([[x,y]]), axis=0)
    def get_coordinates2(event, x, y, flags, param):
        global color_index2, points_dest
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            # Draw a dot with the current rainbow color on the second image
            cv2.circle(im_dst, (x, y), 6, rainbow_colors[color_index2], -1)
            # Move to the next color in the list
            color_index2 = (color_index2 + 1) % len(rainbow_colors)
            points_dest = np.append(points_dest, np.array([[x,y]]), axis=0)

    folder_path = "./img_data"
    images = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    for image_path in images:
    # Read the image
        im_src = cv2.imread(os.path.join(folder_path, image_path))
        im_dst = cv2.imread('ucla.png')
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', get_coordinates)
        cv2.namedWindow('Image2')
        cv2.setMouseCallback('Image2', get_coordinates2)

        while True:
    # Display the image
            cv2.imshow('Image', im_src)
            cv2.imshow('Image2', im_dst)

        # Break the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    
    # Four corners of the book in source image
    # pts_src = np.array([[288, 235], [314, 528], [681, 801],[1005, 765], [761, 217], [1656, 751], [1990, 455]])

    # # Read destination image.
    # im_dst = cv2.imread('ucla.png')
    # # Four corners of the book in destination image.
    # pts_dst = np.array([[1304, 26],[1305, 408],[1403, 665],[1497, 665], [1495, 26], [1693, 666], [1888, 405]])

    # Calculate Homography
        h, status = cv2.findHomography(points_src, points_dest)
        print(h)

        flattened_array = h.flatten()
        column_labels = [f'Column{i+1}' for i in range(flattened_array.shape[0])]
        df = pd.DataFrame([flattened_array], columns=column_labels)
        df.insert(0,'Filename',image_path)
        df.to_csv(csv_file, mode='a', header=False, index=False)

    # # Warp source image to destination based on homography
        im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

    # points_to_transform = np.array([[0, 221]], dtype=float)
    # transformed_points = cv2.perspectiveTransform(np.array([points_to_transform]), h)
    # print(transformed_points)

    # Display images
        cv2.imshow("Source Image", im_src)
        cv2.imshow("Destination Image", im_dst)
        cv2.imshow("Warped Source Image", im_out)
        points_src = np.empty((0, 2), int)
        points_dest = np.empty((0, 2), int)
        cv2.waitKey(0)