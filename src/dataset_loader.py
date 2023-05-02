# Default libraries
import os

import torch
# Require installation
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CatDatasetLoader(Dataset):
    def __init__(self, dataset: str, rescale_size: tuple[int, int], data_path: str = '/data'):
        super().__init__()

        # define path & basic parameters
        self.path = os.path.join(data_path, dataset)
        self.rescale_size = rescale_size

        # read images folder
        self.files = os.listdir(self.path)
        self.len_ = len(self.files)

    def load_image(self, file) -> Image:
        image = Image.open(os.path.join(self.path, file))
        image.load()
        return image

    def __getitem__(self, index: int) -> torch.Tensor:
        transform = transforms.Compose([
            # augmentation
            transforms.Resize(self.rescale_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        x = self.load_image(self.files[index])
        x = transform(x)

        return x

    def __len__(self) -> int:
        return self.len_

