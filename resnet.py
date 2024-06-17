import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class HomographyResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(HomographyResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 8)  # Output 8 values for the homography matrix

    def forward(self, x):
        return self.resnet(x)
    
class HomographyDatasetFromCSV(Dataset):
    def __init__(self, csv_file, transform=None):
        df = pd.read_csv(csv_file, header=None)
        # print(df.head())
        # print(df.info())
        df[0] = df[0].apply(lambda x: "/Users/andre/repos/python_hm/img_done/" + str(x))
        self.data = df
        self.image_paths = self.data.iloc[:, 0].values
        self.homographies = self.data.iloc[:, 10:].values
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        homography = self.homographies[idx].astype('float32')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(homography, dtype=torch.float32)
    
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    csv_file = 'data_labels.csv'
    dataset = HomographyDatasetFromCSV(csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HomographyResNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 29
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, homographies in dataloader:
            images, homographies = images.to(device), homographies.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, homographies)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), 'homography_model_corner.pth')