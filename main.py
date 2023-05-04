from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from src.constants import BATCH_SIZE
from src.dataset_loader import CatDataset, parse_dataset
from src.ga import GeneticAlgorithm
from sklearn.model_selection import train_test_split


def sample_from_decoder(model, latent_distribution=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input_size = (1, 3, 32, 32)
    with torch.no_grad():
        img = torch.from_numpy(np.random.random(input_size)).float().to(device)
        latent_sample = model.encoder(img)
    latent_size = latent_sample.detach().cpu().shape
    latent = np.random.random(latent_size)
    latent = torch.from_numpy(latent).float().to(device)
    img = model.decoder(latent).cpu().detach().numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.squeeze(img * 255).astype(np.uint8).transpose(1, 2, 0)
    plt.imshow(img)
    plt.savefig('samples/test1.png')


if __name__ == '__main__':
    # wandb.init(project='GA_training', entity='b21ds01-nic-project')

    files = parse_dataset(dataset='cats')
    train, val = train_test_split(files, train_size=0.8, random_state=42)

    train_data = CatDataset(dataset='cats', rescale_size=(64, 64), do_augmentation=True, files=train)
    val_data = CatDataset(dataset='cats', rescale_size=(64, 64), do_augmentation=False, files=val)
    # ga = GeneticAlgorithm(train_data, val_data, batch_size=BATCH_SIZE)
    # ga_k = 3
    # ga_n_trial = 2
    # epochs_per_sample = 10
    # model, loss = ga.train_ga(k=ga_k, n_trial=ga_n_trial, save_best=False, epochs_per_sample=epochs_per_sample)
    # torch.save(model, Path(f'./checkpoints/model_k{ga_k}_{ga_n_trial}.pth'))
    # model = torch.load('./checkpoints/model_k1_1.pth')
    # sample_from_decoder(model)
    # ga._fit_autoencoder('Tanh conv_3_64_7 conv_64_27_5 conv_27_8_3 conv_8_4_1'.split(), 1)
