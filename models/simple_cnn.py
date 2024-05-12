import torch
import torch.nn as nn
import torch.nn.functional as f


class SimpleCNN(nn.Module):
    """
    Changes from MNIST_CNN:
    - Input channels changed from 1 to 3 for RGB images
    - Input size changed from 28 to 224 for 224x224 images
    - Output channels changed from 10 to 1 for grayscale fixation maps
    - 56 for pooling, as 224/4 = 56 (4 as there are 2 max pools with stride 2)
    - Added sigmoid activation to ensure output is in the range [0, 1]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 120)
        self.fc2 = nn.Linear(120, 1 * 224 * 224)

    def forward(self, xb):
        xb = xb.view(-1, 3, 224, 224)
        xb = f.max_pool2d(f.relu(self.conv1(xb)), 2)
        xb = f.max_pool2d(f.relu(self.conv2(xb)), 2)
        xb = xb.view(-1, 32 * 56 * 56)
        xb = f.relu(self.fc1(xb))
        xb = self.fc2(xb)
        xb = torch.sigmoid(xb)
        return xb.view(-1, 1, 224, 224)
