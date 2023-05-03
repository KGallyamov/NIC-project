# Test libraries
import unittest
import torch.nn as nn
import os
import random

# Test units
from src.model import AutoEncoder, iterate
from src.dataset_loader import CatDatasetLoader


class TestIterate(unittest.TestCase):
    def test_iterate(self):
        answer = [1, 2, 3, 4, 5]

        sample1 = [1, 2, 3, 4, 5]
        self.assertTrue(answer == list(iterate(sample1)), '1d list failed')

        sample2 = [[1, 2], [3, 4], 5]
        self.assertTrue(answer == list(iterate(sample2)), '2d list failed')

        sample3 = [[1, [2, 3]], [[4], 5]]
        self.assertTrue(answer == list(iterate(sample3)), '3d list failed')

        sample4 = [[[[1], 2], 3], 4, [[[5]]]]
        self.assertTrue(answer == list(iterate(sample4)), '4d list failed')

        sample5 = [1, [2, [3, [4, [5]]]]]
        self.assertTrue(answer == list(iterate(sample5)), '5d list failed')


class TestAutoEncoder(unittest.TestCase):
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
    image_shape = (32, 32)

    def test_methods_exist(self):
        methods = dir(AutoEncoder)
        self.assertTrue('forward' in methods, 'Forward is not implemented')

    def test_autoencoder_work(self):
        example = ['ReLU', 'conv_3_32_3', 'conv_32_64_3']
        autoencoder = AutoEncoder(example, self.image_shape)

        self.assertTrue(type(autoencoder) == AutoEncoder, 'AutoEncoder class do now run')
        self.assertTrue(type(autoencoder.encoder) == nn.Sequential, 'Encoder is not defined')
        self.assertTrue(type(autoencoder.decoder) == nn.Sequential, 'Decoder is not defined')

    def test_forward(self):
        dataset = CatDatasetLoader('cifar-10-cats', self.image_shape, data_path=self.data_path)

        example = ['ReLU', 'conv_3_32_3', 'conv_32_64_3', 'linear_128_64', 'linear_64_32']
        autoencoder = AutoEncoder(example, self.image_shape)

        n_trials = 100
        for _ in range(n_trials):
            ind = random.randint(0, len(dataset) - 1)

            result = autoencoder.forward(dataset[ind])
            self.assertTrue(result.size() == dataset[ind].size(), 'Incorrect output shape')


if __name__ == '__main__':
    unittest.main()

