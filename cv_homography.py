import cv2
import numpy as np
 
if __name__ == '__main__' :
    im_src = cv2.imread('test_field.png')
    # Four corners of the book in source image
    pts_src = np.array([[288, 235], [314, 528], [681, 801],[1005, 765], [761, 217], [1656, 751], [1990, 455]])

    # Read destination image.
    im_dst = cv2.imread('ucla.png')
    # Four corners of the book in destination image.
    pts_dst = np.array([[1304, 26],[1305, 408],[1403, 665],[1497, 665], [1495, 26], [1693, 666], [1888, 405]])

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

    points_to_transform = np.array([[0, 221]], dtype=float)
    transformed_points = cv2.perspectiveTransform(np.array([points_to_transform]), h)
    print(transformed_points)

    # Display images
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)
    cv2.waitKey(0)