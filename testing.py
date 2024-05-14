import os

import numpy as np
import torch
import yaml
from PIL import Image

from dataset import get_dataloader
from logger import logger
from models.resnet18 import ResNet18
from models.resnet50 import ResNet50FCN
from models.simple_cnn import SimpleCNN
from utils import get_device


def main():
    # Configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = get_device()
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    checkpoint_file = config['checkpoint_file']
    model_type = config['model_type']

    # Model Initialization
    model_map = {
        'SimpleCNN': SimpleCNN,
        'ResNet18': lambda: ResNet18(num_out_classes=224 * 224),
        'ResNet50FCN': ResNet50FCN,
    }
    model_class = model_map[model_type]
    model = model_class()
    model = model.to(device)

    if checkpoint_file is None:
        raise ValueError("Checkpoint file must be provided for testing.")
    else:
        # Load checkpoint
        checkpoint = torch.load(os.path.join("checkpoints", checkpoint_file))
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    with torch.no_grad():
        # Load test images
        test_loader = get_dataloader(root_dir="data", image_file="data/test_images.txt",
                                     fixation_file=None, batch_size=batch_size,
                                     num_workers=num_workers, shuffle=False)

        for i, batch in enumerate(test_loader):
            logger.info(f'Testing, Batch: [{i + 1}/{len(test_loader)}]')

            images = batch['image'].to(device)

            # Generate fixation map
            pred = model(images)
            pred_map = torch.sigmoid(pred)  # Apply sigmoid to pred_map

            # Convert to grayscale image and save
            for j in range(pred_map.shape[0]):
                img = pred_map[j].squeeze().cpu().numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))  # Convert to 8-bit grayscale
                image_name = os.path.splitext(os.path.basename(batch['image_path'][j]))[0]
                image_name = image_name.replace("image-", "")
                img.save(f"data/fixations/test/prediction-{image_name}.png")


if __name__ == '__main__':
    main()
