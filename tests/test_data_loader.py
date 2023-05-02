# Test libraries
import unittest
import os
import torch

# Test units
from src.dataset_loader import CatDatasetLoader


class TestStringMethods(unittest.TestCase):
    def test_methods_exist(self):
        methods = dir(CatDatasetLoader)
        self.assertTrue('load_image' in methods, 'Loading is not implemented')
        self.assertTrue('__getitem__' in methods, '[ind] is not implemented')
        self.assertTrue('__len__' in methods, 'len() is not implemented')

    def test_correct_location(self):
        dataset = 'cifar-10-cats'
        dataset_path = '../data'

        loader = CatDatasetLoader(dataset, (32, 32), data_path=dataset_path)

        self.assertTrue(len(loader) == len(os.listdir(os.path.join(dataset_path, dataset))), 'Incorrect dataset size')

    def test_index_working(self):
        dataset = 'cifar-10-cats'
        dataset_path = '../data'
        shape_size = (32, 32)

        loader = CatDatasetLoader(dataset, shape_size, data_path=dataset_path)

        self.assertTrue(type(loader[0]) == torch.Tensor, 'Index not working properly')

    def test_shape(self):
        dataset = 'cifar-10-cats'
        dataset_path = '../data'
        shape_size = (32, 32)

        loader = CatDatasetLoader(dataset, shape_size, data_path=dataset_path)

        self.assertTrue(loader[0].size() == [3, *shape_size], 'Reshape now working properly')


if __name__ == '__main__':
    unittest.main()

