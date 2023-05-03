# Default libraries
from typing import List, Tuple, Union
from random import randint, choice, choices, random
from heapq import nlargest  # used for optimization

# Code for GA training is adapted from the labs
from tqdm import tqdm
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils

# Our units
from src.constants import ACTIVATIONS, LINEAR_FEATURES, CONV_FEATURES, KERNEL_SIZE, KERNEL_SIZE_WEIGHTS
from src.model import AutoEncoder


class GeneticAlgorithm:
    def __init__(self, train_data, val_data, batch_size):
        self.train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = data_utils.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.fitness = dict()
        self.data_size = train_data[0].shape

    def mutate(self, x: List[str], p: float) -> List[str]:
        """
        Given config of a single AE, mutate each layer with probability p
        sample_arch = ['ReLU', 'conv_3_32_5', 'conv_32_64_5',
                       'conv_64_128_3', 'linear_4096_1024',
                       'linear_1024_512', 'linear_512_128']
        ga = GeneticAlgorithm([(0, 0)], [(0, 0)], 1)
        print(ga.mutate(sample_arch, p=1.0))
        (non-deterministic)
        >>> ['Sigmoid', 'conv_3_32_5', 'conv_32_70_5', 'conv_70_128_3', 'linear_4096_1024', 'linear_1024_512', 'linear_512_128']
        (non-deterministic)
        >>> ['Tanh', 'conv_3_48_9', 'conv_48_128_3', 'linear_4096_1024', 'linear_1024_512', 'linear_512_128']
        :param x: Original chromosome
        :param p: Mutation chance
        :return: Updated config
        """
        mutated_x = x.copy()
        if np.random.random() < p:  # Change the activation function with probability p
            mutated_x[0] = np.random.choice([act for act in ACTIVATIONS if act != x[0]])
        action_prob = np.random.random()
        # Insert a new layer with probability p / 3
        if action_prob <= p / 3 and len(mutated_x) > 2:
            ind = np.random.randint(1, len(mutated_x) - 1)
            expansion_result = self._expand_layers(mutated_x[ind], mutated_x[ind + 1])
            if expansion_result is not None:
                mutated_x[ind], new_layer, mutated_x[ind + 1] = expansion_result
                mutated_x.insert(ind + 1, new_layer)
            return mutated_x
        # Delete a random layer with probability p / 3
        elif p / 3 < action_prob < 2 * p / 3 and len(mutated_x) > 3:
            ind = np.random.randint(2, len(mutated_x) - 2)
            rm_layer = mutated_x[ind]
            del mutated_x[ind]
            compression_result = self._compress_layers(mutated_x[ind - 1], rm_layer, mutated_x[ind])
            if compression_result is not None:
                mutated_x[ind - 1], mutated_x[ind] = compression_result
            else:
                mutated_x.insert(ind, rm_layer)
            return mutated_x
        # Change the number of neurons in a random layer, again, with probability p / 3
        elif action_prob <= p and len(mutated_x) > 3:
            ind = np.random.randint(2, len(mutated_x))
            alteration_result = self._alter_layer(mutated_x[ind - 1], mutated_x[ind])
            if alteration_result is not None:
                mutated_x[ind - 1], mutated_x[ind] = alteration_result
        return mutated_x

    def maintain_restrictions(self, x: List[str]) -> List[str]:
        '''
        The list of restrictions:
        1. Convolutions strictly before fully connected layers
        2. Gradually decreasing number of features

        :param x:
        :return: individual with applied restrictions
        '''

        # fix restriction 1 via removing any [f,c,f] and [c,f,c] sequences
        rule_1_x = [x[0]]
        for i in range(1, len(x)):
            if rule_1_x[-1].split('_')[0] == 'linear' and x[i].split('_')[0] == 'conv':
                continue
            rule_1_x.append(x[i])

        # fix restriction 2 via removing increasing sequences
        rule_2_x = [rule_1_x[0], rule_1_x[1]]
        min_features = int(rule_1_x[1].split('_')[1])
        for i in range(2, len(rule_1_x)):
            current_features = int(rule_1_x[i].split('_')[1])
            if rule_1_x[i - 1].split('_')[0] == 'conv' and rule_1_x[i].split('_')[0] == 'linear':
                min_features = current_features
                rule_2_x.append(rule_1_x[i])
                continue

            if min_features >= current_features:
                min_features = current_features
                rule_2_x.append(rule_1_x[i])

        return rule_2_x

    def _compress_layers(self, left: str, to_rm: str, right: str) -> Union[Tuple[str, str], None]:
        """
        print(GeneticAlgorithm._compress_layers(None, 'conv_3_32_3', 'conv_32_64_5', 'conv_64_128_3'))
        >>> ('conv_3_48_7', 'conv_48_128_3')
        print(GeneticAlgorithm._compress_layers(None, 'linear_1000_32', 'linear_32_64', 'linear_64_128'))
        >>> ('linear_1000_48', 'linear_48_128')
        :param left: Layer before the layer to remove
        :param to_rm: Layer after to compress
        :param right: Layer after the one to be removed
        :return: None if mutation has failed, update layers configs otherwise
        """
        left_conf, to_rm_conf, right_conf = left.split('_'), to_rm.split('_'), right.split('_')
        if not left_conf[0] == to_rm_conf[0] == right_conf[0]:
            return None
        left_fan_out, right_fan_in = int(left_conf[2]), int(right_conf[1])
        new_left = '_'.join(map(str, [*left_conf[:2], (left_fan_out + right_fan_in) // 2]))
        new_right = '_'.join(map(str, [right.split('_')[0], (left_fan_out + right_fan_in) // 2, right_conf[2]]))

        if 'conv' in left:
            left_kernel_size = int(left_conf[-1])
            to_rm_kernel_size = int(to_rm_conf[-1])
            right_kernel_size = right_conf[-1]
            new_left += '_' + str(left_kernel_size + to_rm_kernel_size - 1)
            new_right += '_' + right_kernel_size
        return new_left, new_right

    def _expand_layers(self, left: str, right: str) -> Union[Tuple[str, str, str], None]:
        """
        print(GeneticAlgorithm._expand_layers(None, 'conv_32_64_5', 'conv_64_128_3'))
        >>> ('conv_32_64_2', 'conv_64_96_4', 'conv_96_128_3')
        print(GeneticAlgorithm._expand_layers(None, 'linear_32_64', 'linear_64_128'))
        >>> ('linear_32_64', 'linear_64_96', 'linear_96_128')
        :param left: Layer after which we plan to insert a new layer
        :param right: Layer before which we plan to insert a new layer
        :return: None if mutation has failed, updated layers configs otherwise
        """
        left_conf, right_conf = left.split('_'), right.split('_')
        if not left_conf[0] == right_conf[0]:
            return None
        middle_neurons = (int(left_conf[2]) + int(right_conf[2])) // 2
        middle_conf = [left_conf[0], int(left_conf[2]), middle_neurons]
        right_conf[1] = str(middle_neurons)
        if 'conv' in left:
            left_kernel_size = int(left_conf[-1])
            if left_kernel_size < 2:
                return None
            middle_conf.append(str(left_kernel_size - left_kernel_size // 2 + 1))
            left_conf[-1] = str(left_kernel_size // 2)
        return '_'.join(map(str, left_conf)), '_'.join(map(str, middle_conf)), '_'.join(map(str, right_conf))

    def _alter_layer(self, preceding: str, layer: str) -> Union[Tuple[str, str], None]:
        """
        print(GeneticAlgorithm._alter_layer(None, 'linear_32_64', 'linear_64_128'))
        (non-deterministic)
        >>> ('linear_32_108', 'linear_108_128')
        print(GeneticAlgorithm._alter_layer(None, 'conv_32_64_5', 'conv_64_128_3'))
        (non-deterministic)
        >>> ('conv_32_41_5', 'conv_41_128_3')
        :param preceding: Layer before the one to be altered
        :param layer: layer to be altered
        :return: None if mutation has failed, updated layers configs otherwise
        """
        preceding_conf, layer_conf = preceding.split('_'), layer.split('_')
        if not preceding_conf[0] == layer_conf[0]:
            return None
        try:
            new_neurons_num = np.random.randint(*sorted([int(preceding_conf[1]), int(layer_conf[2])]))
        except ValueError:
            return None
        preceding_conf[2] = str(new_neurons_num)
        layer_conf[1] = str(new_neurons_num)
        return '_'.join(preceding_conf), '_'.join(layer_conf)

    def crossover(self, x1: List[str], x2: List[str]) -> Tuple[List[str], List[str]]:
        p1 = randint(1, len(x1) - 1)
        p2 = randint(1, len(x2) - 1)

        child1 = self.maintain_restrictions(x1[:p1] + x2[p2:])
        child2 = self.maintain_restrictions(x2[:p2] + x1[p1:])

        return child1, child2

    def _get_nlargest(self, elements: List, k: int, key=lambda a: a):
        return nlargest(k, elements, key=key)  # performs faster than sorting

    def get_elite(self, generation: List[List[str]], k: int) -> List[List[str]]:
        """
        Return "k" most fit samples from the population
        :param generation: List of Chromosomes
        :param k: # of top samples
        :return: List of Chromosomes of length "k"
        """
        return self._get_nlargest(generation, k, key=lambda x: self.fitness.get(tuple(x), -1e9))

    def _generate_population(self, k) -> List[List[str]]:
        """
        :param k: Size of the population
        :return: "k" chromosomes
        """
        flatten_size = 1
        for shape in self.data_size:
            flatten_size *= shape

        population = []
        for _ in range(k):
            individual = [choice(ACTIVATIONS)]
            if random() < 0.5:  # fully linear individual
                n_layers = randint(2, 10)
                features = [flatten_size] + sorted(choices(LINEAR_FEATURES, k=n_layers), reverse=True)
                for i in range(n_layers):
                    individual.append(f"linear_{features[i]}_{features[i + 1]}")

            else:  # fully conv individual
                n_layers = randint(2, 5)
                features = [3] + sorted(choices(CONV_FEATURES, k=n_layers), reverse=True)
                kernel_sizes = sorted(choices(KERNEL_SIZE, weights=KERNEL_SIZE_WEIGHTS, k=n_layers), reverse=True)
                for i in range(n_layers):
                    individual.append(f"conv_{features[i]}_{features[i + 1]}_{kernel_sizes[i]}")
            population.append(individual)

        return population

    def train_ga(self,
                 k: int = 10,
                 n_trial: int = 100,
                 keep_parents: bool = False,
                 patience: int = 3,
                 mutation_p: float = 0.2,
                 epochs_per_sample: int = 50,
                 save_best: bool = False
                 ):
        """
        Genetic Algorithm implementation (maximization)
        :param save_best: Whether save & return best globally or best of last iteration
        :param k: Population size
        :param n_trial: Number of iterations
        :param keep_parents: Elitism
        :param patience: Parameter for early stopping
        :return: The most fit individual after "n_trial"s
        :param mutation_p: Probability of mutation
        :param epochs_per_sample: The number of epochs a sample is trained on
        Future improvement: train all models till convergence (from early stop)
        """
        # Generate initial population
        gen = self._generate_population(k)

        # Calculate the initial fitness
        prev_fitness = self.fitness.get(tuple(gen[0]), -1e9)

        # Best chromosome
        best_chromosome = None
        best_fitness = -1e9

        # Flag to stop if there is no improvements for some generations
        early_stop_flag = patience
        for i in tqdm(range(n_trial), desc='GA pbar'):
            gen = self.get_elite(gen, k)
            gen_fitness = self.fitness.get(tuple(gen[0]), -1e9)

            if best_fitness < gen_fitness:
                best_fitness = gen_fitness
                best_chromosome = gen[0]

            wandb.log({"val_loss": -best_fitness, "step": i})

            early_stop_flag = early_stop_flag - 1 if prev_fitness - gen_fitness >= 0 else patience
            if early_stop_flag == 0:
                print('Early stop')
                break
            prev_fitness = gen_fitness

            next_gen = []

            # Elitism
            if keep_parents:
                next_gen.extend(gen)

            # Cross over
            for i in range(len(gen)):
                for j in range(i + 1, len(gen)):
                    c1, c2 = self.crossover(gen[i], gen[j])
                    next_gen.append(c1)
                    next_gen.append(c2)

                c = self.mutate(gen[i], mutation_p)

                next_gen.append(c)
            gen = next_gen
            # Train autoencoders encoded in current population and save their fitness
            for chromosome in tqdm(gen, leave=False, desc='configs pbar'):
                model, val_loss = self._fit_autoencoder(chromosome, epochs_per_sample)
                prev_fit = self.fitness.get(tuple(chromosome), 1e9)
                self.fitness[tuple(chromosome)] = min(val_loss, prev_fit)

        # Get the best solution
        top_chromosome = self.get_elite(gen, 1)[0] if not save_best else best_chromosome
        top_model, min_loss = self._fit_autoencoder(top_chromosome, epochs_per_sample)
        return top_model, min_loss

    def _fit_autoencoder(self, cfg: List[str], epochs):
        model = AutoEncoder(cfg)
        criterion = nn.MSELoss()
        model = model.to(self._device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        train_losses = []
        val_losses = []
        min_val_loss = np.inf

        for epoch in tqdm(range(epochs), leave=False, desc=' '.join(cfg)):
            model.train()
            train_losses_per_epoch = []
            for i, X_batch in enumerate(self.train_loader):
                optimizer.zero_grad()
                batch = X_batch.to(self._device)
                reconstructed, latent = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                train_losses_per_epoch.append(loss.item())

            train_losses.append(np.mean(train_losses_per_epoch))

            model.eval()
            val_losses_per_epoch = []
            with torch.no_grad():
                for X_batch in self.val_loader:
                    batch = X_batch.to(self._device)
                    reconstructed, latent = model(batch)
                    loss = criterion(reconstructed, batch)
                    val_losses_per_epoch.append(loss.item())
            val_losses.append(np.mean(val_losses_per_epoch))
            if val_losses[-1] <= min_val_loss:
                min_val_loss = val_losses[-1]
                torch.save(model.state_dict(), './models/best_model.pth')

        model.load_state_dict(torch.load('./models/best_model.pth'))
        model.eval()
        return model, min(val_losses)
