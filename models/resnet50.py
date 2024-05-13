import numpy as np
import torch
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import fcn_resnet50
from torch.nn import functional as f, Parameter

from utils import gaussian


class ResNet50FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the FCN with a ResNet-50 backbone
        self.resnet = fcn_resnet50(weights=None, weights_backbone=ResNet50_Weights.DEFAULT, num_classes=1)
        # Optional: Disable the gradients of all parameters in the backbone
        for param in self.resnet.backbone.parameters():
            param.requires_grad = False

        # Create the Gaussian kernel and save it as a non-trainable parameter
        g = gaussian(window_size=25, sigma=11.2)
        self.kernel = Parameter(torch.matmul(g.unsqueeze(-1), g.unsqueeze(-1).t()), requires_grad=False)

        # Load the center bias density, convert it into a tensor, and save it as a non-trainable parameter
        center_bias_density = np.load('data/center_bias_density.npy')
        assert np.isclose(center_bias_density.sum(), 1), "Center bias density is not normalized (does not sum to 1)."
        self.center_bias = Parameter(torch.from_numpy(np.log(center_bias_density)), requires_grad=False)

    def forward(self, xb):
        xb = self.resnet(xb)['out']

        # Reshape the kernel to have four dimensions and apply it to the raw predictions
        kernel = self.kernel[None, None, :, :]
        xb = f.conv2d(xb, kernel, padding=self.kernel.shape[-1] // 2)

        # Add the log center bias density to the smoothed predictions
        xb += self.center_bias

        return xb.view(-1, 1, 224, 224)
