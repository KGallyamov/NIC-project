# Default libraries
import gc
from typing import List, Tuple, Union
from random import randint, choice, choices
from heapq import nlargest  # used for optimization

# Code for GA training is adapted from the labs
from tqdm import tqdm
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils

# Our units
from src.constants import ACTIVATIONS, LINEAR_FEATURES, LATENT_SIZE
from src.model import AutoEncoder


class GeneticAlgorithm:
    def __init__(self, train_data, val_data, batch_size):
        """
        Initialization of GA

        :param train_data:  data to train on
        :param val_data:    validation data
        :param batch_size:  size of batch
        """
        self.train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = data_utils.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.fitness = dict()
        self.data_size = train_data[0].shape

        self.n_runs = 0
        self.skips = []

    def mutate(self, x: List[str], p: float) -> List[str]:
        """
        Given config of a single AE, mutate each layer with probability p

        Examples:
            sample_arch = ['ReLU', 'linear_4096_1024',
                           'linear_1024_512', 'linear_512_128']
            ga = GeneticAlgorithm([(0, 0)], [(0, 0)], 1)
            print(ga.mutate(sample_arch, p=1.0))
            (non-deterministic)
            >>> ['Sigmoid', 'linear_4096_512', 'linear_512_128']
            (non-deterministic)
            >>> ['Tanh', 'linear_4096_2048', 'linear_2048_1024', 'linear_1024_512', 'linear_512_128']


        :param x:  original chromosome
        :param p:  mutation chance
        :return:   updated config
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
        elif p / 3 < action_prob < 2 * p / 3 and len(mutated_x) > 4:
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

    @staticmethod
    def maintain_restrictions(x: List[str]) -> List[str]:
        """
        The list of restrictions:
        1. Gradually decreasing number of features

        :param x:  config that represents autoencoder architecture
        :return:   individual with applied restrictions
        """

        # fix restriction 1 via removing any [f,c,f] and [c,f,c] sequences
        rule_1_x = x.copy()

        # fix restriction 2 via removing increasing sequences
        rule_2_x = [rule_1_x[0], rule_1_x[1]]
        min_features = int(rule_1_x[1].split('_')[2])
        for i in range(2, len(rule_1_x)):
            current_features = int(rule_1_x[i].split('_')[1])

            if min_features >= current_features:
                min_features = current_features
                rule_2_x.append(rule_1_x[i])

        for i in range(2, len(rule_2_x)):
            if rule_2_x[i - 1].split('_')[0] == rule_2_x[i].split('_')[0]:
                gen_prev = rule_2_x[i - 1].split('_')
                gen_next = rule_2_x[i].split('_')
                rule_2_x[i - 1] = '_'.join([gen_prev[0], gen_prev[1], gen_next[1]] + gen_prev[3:])

        return rule_2_x

    @staticmethod
    def _compress_layers(left: str, to_rm: str, right: str) -> Union[Tuple[str, str], None]:
        """
        Transform 3 continious same-type layers to 2 continious same-type layers (with the same shape)

        Examples:
            print(GeneticAlgorithm._compress_layers('linear_1000_32', 'linear_32_64', 'linear_64_128'))
            >>> ('linear_1000_48', 'linear_48_128')


        :param left:   layer before the layer to remove
        :param to_rm:  layer after to compress
        :param right:  layer after the one to be removed
        :return:       None if mutation has failed, update layers configs otherwise
        """
        left_conf, to_rm_conf, right_conf = left.split('_'), to_rm.split('_'), right.split('_')
        if not left_conf[0] == to_rm_conf[0] == right_conf[0]:
            return None
        left_fan_out, right_fan_in = int(left_conf[2]), int(right_conf[1])
        new_left = '_'.join(map(str, [*left_conf[:2], (left_fan_out + right_fan_in) // 2]))
        new_right = '_'.join(map(str, [right.split('_')[0], (left_fan_out + right_fan_in) // 2, right_conf[2]]))

        return new_left, new_right

    @staticmethod
    def _expand_layers(left: str, right: str) -> Union[Tuple[str, str, str], None]:
        """
        Reverse of _compress_layers (check documentation)

        Examples:
            print(GeneticAlgorithm._expand_layers('linear_32_64', 'linear_64_128'))
            >>> ('linear_32_64', 'linear_64_96', 'linear_96_128')


        :param left:   layer after which we plan to insert a new layer
        :param right:  layer before which we plan to insert a new layer
        :return:       None if mutation has failed, updated layers configs otherwise
        """
        left_conf, right_conf = left.split('_'), right.split('_')
        if not left_conf[0] == right_conf[0]:
            return None
        middle_neurons = (int(left_conf[2]) + int(right_conf[2])) // 2
        middle_conf = [left_conf[0], int(left_conf[2]), middle_neurons]
        right_conf[1] = str(middle_neurons)
        return '_'.join(map(str, left_conf)), '_'.join(map(str, middle_conf)), '_'.join(map(str, right_conf))

    @staticmethod
    def _alter_layer(preceding: str, layer: str) -> Union[Tuple[str, str], None]:
        """
        Change one layer (e.x. number of input neurons)

        Examples:
            print(GeneticAlgorithm._alter_layer('linear_32_64', 'linear_64_128'))
            (non-deterministic)
            >>> ('linear_32_108', 'linear_108_128')

        :param preceding:  layer before the one to be altered
        :param layer:      layer to be altered
        :return:           None if mutation has failed, updated layers configs otherwise
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
        """
        Do cross-over between two architectures

        :param x1:  config1 that represents autoencoder architecture
        :param x2:  config2 that represents autoencoder architecture
        :return:    cross-overed x1 and x2 (with maintained restrictions)
        """
        p1 = randint(1, len(x1) - 1)
        p2 = randint(1, len(x2) - 1)

        child1 = self.maintain_restrictions(x1[:p1] + x2[p2:])
        child2 = self.maintain_restrictions(x2[:p2] + x1[p1:])

        return child1, child2

    @staticmethod
    def _get_nlargest(elements: List, k: int, key=lambda a: a):
        """
        Return `k` the largest elements

        :param elements:  list of values
        :param k:         # of top values to return
        :param key:       function to transform elements (default lambda a: a)
        :return:          list of top-k values
        """
        return nlargest(k, elements, key=key)  # performs faster than sorting + slice

    def get_elite(self, generation: List[List[str]], k: int) -> List[List[str]]:
        """
        Return "k" most fit samples from the population

        :param generation:  list of Chromosomes
        :param k:           # of top samples
        :return:            list of top-k chromosomes
        """
        return self._get_nlargest(generation, k, key=lambda x: - self.fitness.get(tuple(x), -1e9))

    def _generate_population(self, k) -> List[List[str]]:
        """
        Generates population of size `k`

        :param k:  size of the population
        :return:   'k' chromosomes
        """
        flatten_size = 1
        for shape in self.data_size:
            flatten_size *= shape

        population = []
        for _ in range(k):
            individual = [choice(ACTIVATIONS)]

            n_layers = randint(3, 6)
            features = [flatten_size] + sorted(choices(LINEAR_FEATURES, k=n_layers), reverse=True)
            for i in range(n_layers):
                individual.append(f"linear_{features[i]}_{features[i + 1]}")
            individual.append(f"linear_{features[-1]}_{LATENT_SIZE}")
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

        :param k:                  population size
        :param n_trial:            number of iterations
        :param keep_parents:       elitism
        :param patience:           parameter for early stopping
        :param mutation_p:         probability of mutation
        :param epochs_per_sample:  the number of epochs a sample is trained on
        :param save_best:          whether save & return best globally or best of last iteration
        :return:                   the most fit individual after "n_trial"s

        Future improvement: train all models till convergence (from early stop)
        """
        # Generate initial population
        gen = self._generate_population(k)

        # Calculate the initial fitness
        prev_fitness = self.fitness.get(tuple(gen[0]), 1e9)

        # Best chromosome
        best_chromosome = None
        best_fitness = 1e9

        # Flag to stop if there is no improvements for some generations
        early_stop_flag = patience
        for i in tqdm(range(n_trial), desc='GA pbar'):
            gen = self.get_elite(gen, k)
            gen_fitness = self.fitness.get(tuple(gen[0]), 1e9)

            if best_fitness >= gen_fitness:
                best_fitness = gen_fitness
                best_chromosome = gen[0]

            if np.abs(best_fitness) < 1e9:
                wandb.log({"val_loss": best_fitness, "step": i})

            early_stop_flag = early_stop_flag - 1 if prev_fitness - gen_fitness >= 0 else patience
            if early_stop_flag == 0:
                print('Early stop in GA')
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
            # Train auto-encoders encoded in current population and save their fitness
            for chromosome in tqdm(gen, leave=False, desc='configs pbar'):
                try:
                    self.n_runs += 1
                    val_loss = self._fit_autoencoder(chromosome, epochs_per_sample)
                except RuntimeError:
                    val_loss = 1e9
                    self.skips.append(chromosome)
                prev_fit = self.fitness.get(tuple(chromosome), 1e9)
                self.fitness[tuple(chromosome)] = min(val_loss, prev_fit)

        self.print_stats()
        # Get the best solution
        top_chromosome = self.get_elite(gen, 1)[0] if not save_best else best_chromosome
        top_model, min_loss, l_means, l_stds = self._fit_autoencoder(top_chromosome, 40, return_model=True)
        return top_model, min_loss, l_means, l_stds

    def print_stats(self):
        print(f'Out of {self.n_runs} runs, {len(self.skips)} was skipped ({len(self.skips) / self.n_runs * 100}%)')
        print(*self.skips, sep='\n')

    def _fit_autoencoder(self, cfg: List[str], epochs, return_model=False) -> Union[tuple[AutoEncoder, float], float]:
        """
        Fit autoencoder with structure `cfg` and train for number of `epochs`

        :param cfg:     list[str] that represents autoencoder structure
        :param epochs:  # of epochs
        :return:        fitted model & minimum val loss
        """
        model = AutoEncoder(cfg, self.data_size[1:])
        criterion = nn.MSELoss()
        model = model.to(self._device)

        optimizer = torch.optim.Adam(model.parameters())

        train_losses = []
        val_losses = []
        min_val_loss = np.inf

        patience = 3
        early_stop_flag = patience

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
            early_stop_flag = early_stop_flag - 1 if min_val_loss < np.mean(val_losses_per_epoch) else patience
            if early_stop_flag == 0:
                break
            if val_losses[-1] <= min_val_loss:
                min_val_loss = val_losses[-1]
                torch.save(model.state_dict(), './models/best_model.pth')
        if return_model:
            model.load_state_dict(torch.load('./models/best_model.pth'))
            model.eval()
            latent_vectors = []
            with torch.no_grad():
                for i, X_batch in enumerate(self.train_loader):
                    optimizer.zero_grad()
                    batch = X_batch.to(self._device)
                    reconstructed, latent = model(batch)
                    lv = latent.detach().cpu().numpy()
                    for j in range(lv.shape[0]):
                        latent_vectors.append(lv[j])
            latent_vectors = np.array(latent_vectors)
            means = latent_vectors.mean(axis=0)
            stds = latent_vectors.std(axis=0)
            return model, np.min(val_losses), means, stds
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return np.min(val_losses)
