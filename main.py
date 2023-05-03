from pathlib import Path

import wandb
import torch
from src.ga import GeneticAlgorithm
from src.constants import BATCH_SIZE
from src.dataset_loader import CatDataset


if __name__ == '__main__':
    wandb.init(project='GA_training', entity='b21ds01-nic-project')
    train_data = CatDataset(dataset='cifar-10-cats', rescale_size=(32, 32), do_augmentation=True)
    val_data = CatDataset(dataset='cifar-10-cats', rescale_size=(32, 32), do_augmentation=False)
    ga = GeneticAlgorithm(train_data, val_data, batch_size=BATCH_SIZE)
    ga_k = 3
    ga_n_trial = 4
    epochs_per_sample = 5
    model, loss = ga.train_ga(k=ga_k, n_trial=ga_n_trial, save_best=True, epochs_per_sample=epochs_per_sample)
    torch.save(model, Path(f'./checkpoints/model_k{ga_k}_{ga_n_trial}.pth'))
