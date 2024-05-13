import random

from torchvision.transforms import functional as f
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import imageio.v2 as imageio
import os


def read_text_file(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Or some other preprocessing
            lines.append(line)
    return lines


def get_dataloader(root_dir, image_file, fixation_file, batch_size=16, num_workers=1, shuffle=True):
    # Transforms
    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    fixation_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Instance of the dataset
    dataset = FixationDataset(root_dir=root_dir,
                              image_file=image_file,
                              fixation_file=fixation_file,
                              image_transform=image_transform,
                              fixation_transform=fixation_transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, persistent_workers=True)

    return dataloader


class RandomFlipPair(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, fixation = sample['image'], sample['fixation']
        if random.random() < self.p:
            image = f.hflip(image)
            fixation = f.hflip(fixation)
        if random.random() < self.p:
            image = f.vflip(image)
            fixation = f.vflip(fixation)
        return {'image': image, 'fixation': fixation}


class FixationDataset(Dataset):
    def __init__(self, root_dir, image_file, fixation_file, image_transform=None, fixation_transform=None):
        self.root_dir = root_dir
        self.image_files = read_text_file(image_file)
        self.fixation_files = read_text_file(fixation_file)
        self.image_transform = image_transform
        self.fixation_transform = fixation_transform
        assert len(self.image_files) == len(
            self.fixation_files), "lengths of image files and fixation files do not match!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = imageio.imread(img_name)

        fix_name = os.path.join(self.root_dir, self.fixation_files[idx])
        fix = imageio.imread(fix_name)

        if self.image_transform:
            image = self.image_transform(image)
        if self.fixation_transform:
            fix = self.fixation_transform(fix)

        sample = {"image": image, "fixation": fix}

        return RandomFlipPair()(sample)
