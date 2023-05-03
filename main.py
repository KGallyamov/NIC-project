from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from src.constants import BATCH_SIZE
from src.dataset_loader import CatDataset
from src.ga import GeneticAlgorithm


def sample_from_decoder(model, latent_distribution=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    latent = np.random.random((1, 12, 22, 22))
    latent = torch.from_numpy(latent).float().to(device)
    img = model.decoder(latent).cpu().detach().numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.squeeze(img * 255).astype(np.uint8).transpose(1, 2, 0)
    plt.imshow(img)
    plt.savefig('samples/test.png')


if __name__ == '__main__':
    wandb.init(project='GA_training', entity='b21ds01-nic-project')
    train_data = CatDataset(dataset='cifar-10-cats', rescale_size=(32, 32), do_augmentation=True)
    val_data = CatDataset(dataset='cifar-10-cats', rescale_size=(32, 32), do_augmentation=False)
    ga = GeneticAlgorithm(train_data, val_data, batch_size=BATCH_SIZE)
    ga_k = 3
    ga_n_trial = 2
    epochs_per_sample = 10
    model, loss = ga.train_ga(k=ga_k, n_trial=ga_n_trial, save_best=False, epochs_per_sample=epochs_per_sample)
    torch.save(model, Path(f'./checkpoints/model_k{ga_k}_{ga_n_trial}.pth'))
    model = torch.load(Path(f'./models/best_model.pth'))
    sample_from_decoder(model)
