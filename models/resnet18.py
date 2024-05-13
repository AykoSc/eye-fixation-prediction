import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn.functional as f


class ResNet18(nn.Module):
    def __init__(self, num_out_classes):
        super().__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Load a pre-trained ResNet
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze the parameters of the ResNet
        num_ftrs = self.resnet.fc.in_features  # Replace the fully connected layer
        self.resnet.fc = nn.Linear(num_ftrs, 512)
        self.fc1 = nn.Linear(512, 512)  # New fully connected layer
        self.fc2 = nn.Linear(512, num_out_classes)

    def forward(self, xb):
        xb = self.resnet(xb)
        xb = xb.view(xb.size(0), -1)  # Flatten the output of the ResNet
        xb = self.resnet.fc(xb)
        xb = f.relu(self.fc1(xb))  # Apply ReLU activation function
        xb = self.fc2(xb)

        return xb.view(-1, 1, 224, 224)
