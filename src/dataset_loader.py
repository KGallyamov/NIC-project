# Default libraries
import os

# Require installation
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CatDataset(Dataset):
    def __init__(self, dataset: str, rescale_size: tuple[int, int], data_path: str = 'data', do_augmentation: bool = True):
        """
        Initialization of DatasetLoader

        :param dataset:          name of dataset in `data` folder
        :param rescale_size:     tuple with required rescale_size
        :param data_path:        path of `data` folder (by default we suppose that is the same level as main.py)
        :param do_augmentation:  boolean parameter that identify whether augmentation is required (for validation no)
        """
        super().__init__()

        # define path & basic parameters
        self.path = os.path.join(data_path, dataset)
        self.rescale_size = rescale_size
        self.do_augmentation = do_augmentation

        # read images folder
        self.files = os.listdir(self.path)
        self.len_ = len(self.files)

        # define augmentation
        self.transform = transforms.Compose([
            transforms.Resize(self.rescale_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.resize = transforms.Compose([
            transforms.Resize(self.rescale_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_image(self, file) -> Image:
        """
        Load image from `data/dataset/file`

        :param file:  name of a file (image) to load
        :return:      parsed image
        """
        image = Image.open(os.path.join(self.path, file))
        image.load()
        return image

    def change_do_augmentation(self):
        """
        Change class to do/not do augmentation
        """
        self.do_augmentation = not self.do_augmentation

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Get image by index

        :param index:  index of an image (should be between [0, len(CatDatasetLoader))
        :return:       preprocessed image
        """
        assert 0 <= index < self.len_
        x = self.load_image(self.files[index])
        x = self.transform(x) if self.do_augmentation else self.resize(x)

        return x

    def __len__(self) -> int:
        """
        Get amount of pictures in dataset

        :return:  amount of pictures
        """
        return self.len_

