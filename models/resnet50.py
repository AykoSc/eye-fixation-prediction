import torch
from torchvision.models.segmentation import fcn_resnet50


class ResNet50FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the FCN with a ResNet-50 backbone
        self.resnet = fcn_resnet50(pretrained=False, pretrained_backbone=True, num_classes=1)
        # Optional: Disable the gradients of all parameters in the backbone
        for param in self.resnet.backbone.parameters():
            param.requires_grad = False

    def forward(self, xb):
        xb = self.resnet(xb)['out']
        xb = torch.sigmoid(xb)
        return xb.view(-1, 1, 224, 224)
