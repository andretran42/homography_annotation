import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
from numpy.linalg import inv
import os

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
image_path = './img_pred/img_pred3.png'
image = Image.open(image_path).convert('RGB')
im_dst = cv2.imread('ucla.png')
input_image = transform(image).unsqueeze(0)

# Preprocess the image
# image = transform(image).unsqueeze(0)  # Add batch dimension
width, height = image.size

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HomographyResNet()
model.load_state_dict(torch.load('homography_model_corner.pth', map_location=device))
# model.to(device)
model.eval()

# Pass the image through the model
with torch.no_grad():
    input_image = input_image.to(device)
    output = model(input_image)

# Process the output if necessary
predicted_homography = output.cpu().numpy()[0]  # Convert to numpy array and move to CPU if using GPU

# Now you can use the predicted_homography as needed
print(predicted_homography)
image = cv2.imread(image_path)
dim = image.shape
p1 = np.array([[0, 0]], dtype=float) #top left
p2 = np.array([[dim[1], 0]], dtype=float) #top right
p3 = np.array([[dim[1], dim[0]]], dtype=float) #bottom right
p4 = np.array([[0, dim[0]]], dtype=float) #bottom left
wp1 = np.array([predicted_homography[0], predicted_homography[1]])
wp2 = np.array([predicted_homography[2], predicted_homography[3]])
wp3 = np.array([predicted_homography[4], predicted_homography[5]])
wp4 = np.array([predicted_homography[6], predicted_homography[7]])
# [ 859.1184  -441.1654  1009.2742  -230.58026  973.4259   854.93225
#   909.9304   822.71655]
# wp1 = np.array([1205, -20], dtype=float)
# wp2 = np.array([1747, 85], dtype=float)
# wp3 = np.array([1500, 854], dtype=float)
# wp4 = np.array([1229, 822], dtype=float)
print(dim[1], dim[0])
pts_src = np.array([p1, p2, p3, p4])
pts_dest = np.array([wp1, wp2, wp3, wp4])
hm, status = cv2.findHomography(pts_src, pts_dest)
hm_inv = inv(hm)
print(hm)
im_out = cv2.warpPerspective(image, hm, (im_dst.shape[1],im_dst.shape[0]))
# im_out = cv2.warpPerspective(im_dst, hm_inv, (image.shape[1], image.shape[0]))


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

# Display the original and warped images
cv2.imshow('Original Image', image)
cv2.imshow('Warped Image', im_out)

# Wait for a key press and close the image windows
cv2.waitKey(0)
cv2.destroyAllWindows()