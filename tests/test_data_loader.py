# Test libraries
import unittest
import os
import torch

# Test units
from src.dataset_loader import CatDatasetLoader


class TestDataLoader(unittest.TestCase):
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')

    def test_methods_exist(self):
        methods = dir(CatDatasetLoader)
        self.assertTrue('load_image' in methods, 'Loading is not implemented')
        self.assertTrue('__getitem__' in methods, '[ind] is not implemented')
        self.assertTrue('__len__' in methods, 'len() is not implemented')

    def test_correct_location(self):
        dataset = 'cifar-10-cats'

        loader = CatDatasetLoader(dataset, (32, 32), data_path=self.data_path)

        self.assertTrue(len(loader) == len(os.listdir(os.path.join(self.data_path, dataset))), 'Incorrect dataset size')

    def test_index_working(self):
        dataset = 'cifar-10-cats'
        shape_size = (32, 32)

        loader = CatDatasetLoader(dataset, shape_size, data_path=self.data_path)

        self.assertTrue(type(loader[0]) == torch.Tensor, 'Index not working properly')

    def test_shape(self):
        dataset = 'cifar-10-cats'

        shape_size = (32, 32)
        loader = CatDatasetLoader(dataset, shape_size, data_path=self.data_path)
        self.assertTrue(loader[0].size() == [3, *shape_size], 'Reshape now working properly')

        shape_size = (10, 10)
        loader = CatDatasetLoader(dataset, shape_size, data_path=self.data_path)
        self.assertTrue(loader[0].size() == [3, *shape_size], 'Reshape now working properly')

    def test_do_augmentation(self):
        dataset = 'cifar-10-cats'

        shape_size = (32, 32)
        loader = CatDatasetLoader(dataset, shape_size, data_path=self.data_path)

        before = loader.do_augmentation
        loader.change_do_augmentation()
        after = loader.do_augmentation

        self.assertTrue(before != after, 'Augmentation did not changed')


if __name__ == '__main__':
    unittest.main()

