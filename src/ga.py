import numpy as np
from typing import List, Tuple
from random import randint

import torch
import torch.nn as nn

from tqdm import tqdm
import torch.utils.data as data_utils
from model import AutoEncoder


class GeneticAlgorithm:

    def __init__(self, train_data, val_data, batch_size):
        self.train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = data_utils.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

    def mutate(self, x: List[str]) -> List[str]:
        pass

    def crossover(self, x1: List[str], x2: List[str]) -> Tuple[List[str], List[str]]:
        p1 = randint(1, len(x1) - 1)
        p2 = randint(1, len(x2) - 1)

        child1 = self.maintain_restrictions(x1[:p1] + x2[p2:])
        child2 = self.maintain_restrictions(x2[:p2] + x1[p1:])

        return child1, child2

    def train_ga(self, steps=100, population_size=20, epochs_per_sample=50):
        pass

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
