
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

class ScenicCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(ScenicCNN, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x) * 10

def load_scenic_model(model_path='scenic_cnn_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    model = ScenicCNN()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return model, transform, device

def score_image(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')

    with torch.no_grad():
        img_tensor = transform(image).unsqueeze(0).to(device)
        output = model(img_tensor)
        score = output.squeeze().cpu().item()
        return max(0, min(10, score))

# Example usage:
# model, transform, device = load_scenic_model()
# score = score_image('path/to/tile.png', model, transform, device)
