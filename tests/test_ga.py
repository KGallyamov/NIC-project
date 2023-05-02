# Test libraries
import unittest
import os

# Test units
from src.ga import GeneticAlgorithm
from src.dataset_loader import CatDatasetLoader


class TestGeneticAlgorithm(unittest.TestCase):
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')

    def test_methods_exist(self):
        methods = dir(GeneticAlgorithm)
        self.assertTrue('mutate' in methods, 'Mutations is not implemented')
        self.assertTrue('crossover' in methods, 'Crossover is not implemented')
        self.assertTrue('train_ga' in methods, 'Training is not implemented')
        self.assertTrue('compute_fitness' in methods, 'Fitness is not implemented')
        self.assertTrue('_generate_population' in methods, 'Population generation is not implemented')

    def test_get_elite(self):
        dataset = CatDatasetLoader('cifar-10-cats', (32, 32), data_path=self.data_path)
        ga = GeneticAlgorithm(dataset, dataset, 1)

        self.assertTrue(ga._get_nlargest([], 10) == [], 'Empty list failure')
        self.assertTrue(ga._get_nlargest([4, 1, 3, 5, 9, 2, 7, 1, 6, 8, 10], 5) == [10, 9, 8, 7, 6], 'Unique numbers')
        self.assertTrue(ga._get_nlargest([5, 5, 3, 3, 1, 1, 2, 2, 8, 8, 4, 4, 9, 9, 0, 0], 2) == [9, 9], 'Same elements')
        self.assertTrue(ga._get_nlargest([1, 2, 3, 4, 5, 6, 7], 3, key=lambda a: -a) == [1, 2, 3], 'Minimization key')
        self.assertTrue(ga._get_nlargest([1.56, -0.4, 2, -6, 0.9, 3], 2, key=lambda a: a**2) == [-6, 3], 'Float numbers')

    def test_mutation(self):
        dataset = CatDatasetLoader('cifar-10-cats', (32, 32), data_path=self.data_path)
        ga = GeneticAlgorithm(dataset, dataset, 1)

        # Check that probability is working properly (by setting probability to 1)
        n_tries = 100
        tries_results_1 = []
        tries_results_0 = []
        sample1 = ['ReLU', 'linear_128_64', 'linear_64_32']

        for _ in range(n_tries):
            tries_results_1.append(ga.mutate(sample1, 1))
            tries_results_0.append(ga.mutate(sample1, 0))

        self.assertTrue(all(mutated != sample1 for mutated in tries_results_1), 'Mutation probability 1 did not change sample')
        self.assertTrue(all(mutated == sample1 for mutated in tries_results_0), 'Mutation probability 0 did change sample')

        # Check that all types of mutations are present
        n_tries = 100
        tries_results = []
        sample2 = ['Sigmoid', 'conv_3_32_5', 'conv_32_70_5', 'conv_70_128_3', 'linear_4096_1024', 'linear_1024_512', 'linear_512_128']

        for _ in range(n_tries):
            tries_results.append(ga.mutate(sample2, 1))

        self.assertTrue(any(len(mutated) > len(sample2) for mutated in tries_results), 'No increase in size')
        self.assertTrue(any(len(mutated) < len(sample2) for mutated in tries_results), 'No decrease in size')
        self.assertTrue(any(mutated[1:] != sample2[1:] for mutated in tries_results if len(mutated) == len(sample2)), 'No change in number of neurons')


if __name__ == '__main__':
    unittest.main()

