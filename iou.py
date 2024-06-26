import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import os
from shapely.geometry import box, Polygon
from numpy.linalg import inv

class HomographyResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(HomographyResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 8)  # Output 8 values for the homography matrix

    def forward(self, x):
        return self.resnet(x)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
image_path = './img_pred/img_pred5.png'
image = Image.open(image_path).convert('RGB')
im_dst = cv2.imread('ucla.png')
input_image = transform(image).unsqueeze(0)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HomographyResNet()
model.load_state_dict(torch.load('homography_model_corner.pth', map_location=device))
# model.to(device)
model.eval()


# Process the output if necessary
  # Convert to numpy array and move to CPU if using GPU

total_sum = 0
total_num = 0

df = pd.read_csv('./test_labels.csv', header=None)
for index, row in df.iterrows():
    im_path = "./img_pred/" + row[0]
    image = Image.open(im_path).convert('RGB')
    input_image = transform(image).unsqueeze(0)

    with torch.no_grad():
        input_image = input_image.to(device)
        output = model(input_image)

    predicted_homography = output.cpu().numpy()[0]

    image = cv2.imread(im_path)
    dim = image.shape
    p1 = [row[10], row[11]]
    p2 = [row[12], row[13]]
    p3 = [row[14], row[15]]
    p4 = [row[16], row[17]]
    wp1 = [predicted_homography[0], predicted_homography[1]]
    wp2 = [predicted_homography[2], predicted_homography[3]]
    wp3 = [predicted_homography[4], predicted_homography[5]]
    wp4 = [predicted_homography[6], predicted_homography[7]]
    ap1 = np.array([p1], dtype=float) #top left
    ap2 = np.array([p2], dtype=float) #top right
    ap3 = np.array([p3], dtype=float) #bottom right
    ap4 = np.array([p4], dtype=float) #bottom left
    awp1 = np.array([wp1], dtype=float)
    awp2 = np.array([wp2], dtype=float)
    awp3 = np.array([wp3], dtype=float)
    awp4 = np.array([wp4], dtype=float)
    pts_src = np.array([ap1, ap2, ap3, ap4])
    pts_dest = np.array([awp1, awp2, awp3, awp4])
    # hm, status = cv2.findHomography(pts_src, pts_dest)
    # wwp1 = cv2.perspectiveTransform(np.array([awp1]), hm).reshape(2)
    # wwp2 = cv2.perspectiveTransform(np.array([awp2]), hm).reshape(2)
    # wwp3 = cv2.perspectiveTransform(np.array([awp3]), hm).reshape(2)
    # wwp4 = cv2.perspectiveTransform(np.array([awp4]), hm).reshape(2)
    # wap1 = cv2.perspectiveTransform(np.array([ap1]), hm).reshape(2)
    # wap2 = cv2.perspectiveTransform(np.array([ap2]), hm).reshape(2)
    # wap3 = cv2.perspectiveTransform(np.array([ap3]), hm).reshape(2)
    # wap4 = cv2.perspectiveTransform(np.array([ap4]), hm).reshape(2)

    # pol1_xy = [wap1, wap2, wap3, wap4]
    # pol2_xy = [wwp1, wwp2, wwp3, wwp4]
    pol1_xy = [p1, p2, p3, p4]
    pol2_xy = [wp1, wp2, wp3, wp4]
    polygon1_shape = Polygon(pol1_xy)
    polygon2_shape = Polygon(pol2_xy)

    # Calculate Intersection and union, and tne IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.union(polygon2_shape).area
    IOU = polygon_intersection / polygon_union 

    total_sum += IOU
    total_num += 1
    print(str(IOU) + ": " + row[0])
print(total_sum/total_num)



# homography_matrix = np.array([
#     [predicted_homography[0],predicted_homography[1],predicted_homography[2]],  # Example values
#     [predicted_homography[3],predicted_homography[4],predicted_homography[5]],
#     [predicted_homography[6],predicted_homography[7], 1.0]
# ], dtype=np.float32)

# homography_matrix = np.array([
#     [2,0,0],  # Example values
#     [1,1,1],
#     [0,0, 1.0]
# ], dtype=np.float32)
# image_cv = cv2.imread(image_path)
# im_dst = cv2.imread('ucla.png')
# height, width = im_dst.shape[:2]
# warped_image = cv2.warpPerspective(image_cv, homography_matrix, (width, height))

# # Display the original and warped images
# cv2.imshow('Original Image', image)
# cv2.imshow('Warped Image', im_out)

# # Wait for a key press and close the image windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()