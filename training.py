import os

import torch
import yaml
from PIL import Image
from torch import optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

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
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    num_workers = config['num_workers']
    checkpoint_file = config['checkpoint_file']
    model_type = config['model_type']
    show_val_loss = config['show_val_loss']
    show_last_image = config['show_last_image']
    #config states epochs = 10 (In config yaml)

    # Data Loading
    train_loader = get_dataloader(root_dir="data", image_file="data/train_images.txt",
                                  fixation_file="data/train_fixations.txt", batch_size=batch_size,
                                  num_workers=num_workers)
    val_loader = get_dataloader(root_dir="data", image_file="data/val_images.txt",
                                fixation_file="data/val_fixations.txt", batch_size=batch_size * 2,
                                num_workers=num_workers, shuffle=False)

    # Model Initialization
    model_map = {
        'SimpleCNN': SimpleCNN,
        'ResNet18': lambda: ResNet18(num_out_classes=224 * 224),
        'ResNet50FCN': ResNet50FCN,
    }
    model_class = model_map[model_type]
    model = model_class()
    model = model.to(device)

    opt = optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = BCEWithLogitsLoss()
    if checkpoint_file is None:
        start_epoch = 1
        loss = None
    else:
        # Load checkpoint
        checkpoint = torch.load(os.path.join("checkpoints", checkpoint_file))
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']

    # Training
    writer = SummaryWriter('runs')  # For metrics visualization
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            fixations = batch['fixation'].to(device)

            # Forward pass
            pred = model(images)
            loss = loss_func(pred, fixations)

            if show_last_image and i == len(train_loader) - 1:
                # Load the original image
                image_path = batch['image_path'][0]
                image_unnormed = Image.open(image_path)

                pred_map = torch.sigmoid(pred)  # Apply sigmoid to pred_map

                # Make grids
                image_grid = make_grid(transforms.ToTensor()(image_unnormed))
                pred_map_grid = make_grid(pred_map)
                fixation_map_grid = make_grid(fixations)

                # Add images to TensorBoard
                writer.add_image('Image', image_grid, global_step=epoch)
                writer.add_image('Predicted Fixation Map', pred_map_grid, global_step=epoch)
                writer.add_image('Target Fixation Map', fixation_map_grid, global_step=epoch)

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            if loss is not None:
                logger.info(
                    f'Epoch [{epoch}/{num_epochs}], Batch: [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')
                writer.add_scalar('training loss', loss.item(), global_step=epoch * len(train_loader) + i)

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss.item(),
        }, os.path.join("checkpoints", f'{model_type}_{epoch}.pt'))

        model.eval()
        if show_val_loss:
            with torch.no_grad():
                valid_loss = sum(
                    loss_func(model(batch['image'].to(device)), batch['fixation'].to(device))
                    for batch in val_loader
                )
            logger.info(f'Epoch [{epoch}/{num_epochs}], Validation Loss: {valid_loss / len(val_loader)}')
    writer.close()


if __name__ == '__main__':
    main()
