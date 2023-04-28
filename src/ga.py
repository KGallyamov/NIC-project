# Code for GA training is adapted from the labs
import numpy as np
from typing import List, Tuple, Union
from random import randint

import torch
import torch.nn as nn

from tqdm import tqdm
import torch.utils.data as data_utils
from model import AutoEncoder, ACTIVATIONS
from chromosome import Chromosome


class GeneticAlgorithm:

    def __init__(self, train_data, val_data, batch_size):
        self.train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = data_utils.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.fitness = dict()

    def mutate(self, x: List[str], p: float) -> List[str]:
        """
        Given config of a single AE, mutate each layer with probability p
        :param x: Original chromosome
        :param p: Mutation chance
        :return: Updated config
        """
        mutated_x = x.copy()
        if np.random.random() < p:  # Change the activation function with probability p
            mutated_x[0] = np.random.choice([act for act in ACTIVATIONS if act != x[0]])
        action_prob = np.random.random()
        # Insert a new layer with probability p / 3
        if 0 <= action_prob <= p / 3 and len(mutated_x) > 2:
            ind = np.random.randint(1, len(mutated_x) - 1)
            mutated_x[ind], new_layer, mutated_x[ind + 1] = self._expand_layers(mutated_x[ind], mutated_x[ind + 1])
            mutated_x.insert(ind, new_layer)
            return mutated_x
        # Delete a random layer with probability p / 3
        elif p / 3 < action_prob < 2 * p / 3 and len(mutated_x) > 3:
            ind = np.random.randint(2, len(mutated_x) - 2)
            rm_layer = mutated_x[ind]
            del mutated_x[ind]
            mutated_x[ind - 1], mutated_x[ind] = self._compress_layers(mutated_x[ind - 1],
                                                                       rm_layer, mutated_x[ind])
            return mutated_x
        # Change the number of neurons in a random layer, again, with probability p / 3
        elif action_prob < p:
            ind = np.random.randint(1, len(mutated_x))
            mutated_x[ind] = self._alter_layer(mutated_x[ind])
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
        rule_2_x = [rule_1_x[0]]
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

    def _compress_layers(self, left: str, to_rm: str, right: str) -> Union[Tuple[str, str], bool]:
        left_conf, to_rm_conf, right_conf = left.split('_'), to_rm.split('_'), right.split('_')
        if not left_conf[0] == to_rm_conf[0] == right_conf[0]:
            return False
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

    def _expand_layers(self, l1: str, l2: str) -> Tuple[str, str, str]:
        return None

    def _alter_layer(self, layer: str) -> str:
        return None

    def crossover(self, x1: List[str], x2: List[str]) -> Tuple[List[str], List[str]]:
        p1 = randint(1, len(x1) - 1)
        p2 = randint(1, len(x2) - 1)

        child1 = self.maintain_restrictions(x1[:p1] + x2[p2:])
        child2 = self.maintain_restrictions(x2[:p2] + x1[p1:])

        return child1, child2

    def compute_fitness(self, *args, **kwargs) -> float:
        """
        This function should be called only once for each model to optimize performance
        :param args:
        :param kwargs:
        :return:
        """
        return 0

    def get_elite(self, generation: List[List[str]], k) -> List[List[str]]:
        """
        Return "k" most fit samples from the population
        :param generation: List of Chromosomes
        :param k: # of top samples
        :return: List of Chromosomes of length "k"
        """
        generation = sorted(generation, key=lambda x: self.fitness[x])
        return generation[-k:]

    def _generate_population(self, k) -> List[List[str]]:
        """
        :param k: Size of the population
        :return: "k" chromosomes
        """
        return None

    def train_ga(self, k=30, n_trial=100, keep_parents=False, patience=3, mutation_p=0.2, epochs_per_sample=50):
        """
        Genetic Algorithm implementation
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
        prev_fitness = self.fitness[gen[0]]

        # Flag to stop if there is no improvements for some generations
        early_stop_flag = patience
        for _ in tqdm(range(n_trial)):
            gen = self.get_elite(gen, k)
            gen_fitness = self.fitness[gen[-1]]
            # print('\nImprovement', gen_fitness - prev_fitness, 'New score:', gen_fitness)

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
            for chromosome in gen:
                model, tr_history, vl_history = self._fit_autoencoder(chromosome, epochs_per_sample)
                if chromosome not in self.fitness.keys():
                    self.fitness[chromosome] = self.compute_fitness(model)
                else:
                    self.fitness[chromosome] = min(self.compute_fitness(model), self.fitness[chromosome])
        # Get the best solution
        top_chromosome = self.get_elite(gen, 1)[0]
        top_model, top_model_train_losses, top_model_val_losses = self._fit_autoencoder(top_chromosome,
                                                                                        epochs_per_sample)
        return self.fitness[top_chromosome], top_model, top_model_train_losses, top_model_val_losses

    def _fit_autoencoder(self, cfg: List[str], epochs):
        model = AutoEncoder(cfg)
        criterion = nn.MSELoss()
        model = model.to(self._device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        train_losses = []
        val_losses = []
        min_val_loss = np.inf

        for epoch in range(epochs):
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
        return model, train_losses, val_losses


if __name__ == '__main__':
    print(GeneticAlgorithm._compress_layers(None, 'conv_3_32_3', 'conv_32_64_5', 'conv_64_128_3'))
    # >>> ('conv_3_48_7', 'conv_48_128_3')
