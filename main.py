from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from src.constants import BATCH_SIZE
from src.dataset_loader import CatDataset, parse_dataset
from src.ga import GeneticAlgorithm
from sklearn.model_selection import train_test_split


def sample_from_decoder(model, image_shape, means, stds):
    """
    Assuming latent vector components are independent normal r.v,
    sample from the distribution of latent vectors recorded while training
    Saves the generated image to "samples/sample.png"

    :param model:        autoencoder to sample from
    :param image_shape:  output image shape
    :param means:        vector of means of latent vectors
    :param stds:         vector of stds of latent vectors
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    n_images = 16
    model.to(device)

    # calculate size of input image after flatten
    flatten_shape = 1
    for sh in image_shape:
        flatten_shape *= sh
    input_size = (n_images, flatten_shape)

    with torch.no_grad():
        img = torch.from_numpy(np.random.normal(loc=np.zeros(input_size),
                                                scale=np.ones(input_size),
                                                size=input_size)).float().to(device)
        latent_sample = model.encoder(img)

    # Get latent size and make random samples with the same size
    latent_size = latent_sample.detach().cpu().shape
    print(latent_size)
    latent = np.random.normal(loc=means, scale=stds, size=latent_size)
    latent = torch.from_numpy(latent).float().to(device)

    # Plot diverse images
    imgs = model.decoder(latent).cpu().detach().numpy()
    plt_shape = int(np.sqrt(n_images))
    for i in range(n_images):
        img = imgs[i]
        img = np.reshape(img, image_shape)
        img = np.squeeze(img).transpose(1, 2, 0)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        plt.subplot(plt_shape, plt_shape, i + 1)
        plt.axis('off')
        plt.imshow(img)
    plt.savefig('samples/sample.png')


if __name__ == '__main__':
    # initialize wandb (make sure you logged in before running `wandb login`)
    wandb.init(project='GA_training', entity='b21ds01-nic-project')

    # initialize image shape & load dataset
    im_shape = (64, 64)
    files = parse_dataset(dataset='cats')
    train, val = train_test_split(files, train_size=0.8, random_state=42)

    train_data = CatDataset(dataset='cats', rescale_size=im_shape, do_augmentation=True, files=train)
    val_data = CatDataset(dataset='cats', rescale_size=im_shape, do_augmentation=False, files=val)

    # GA itself
    ga = GeneticAlgorithm(train_data, val_data, batch_size=BATCH_SIZE)
    ga_k = 4
    ga_n_trial = 5
    epochs_per_sample = 1
    model, loss, means, stds = ga.train_ga(k=ga_k, n_trial=ga_n_trial, save_best=False,
                                           epochs_per_sample=epochs_per_sample, patience=5)

    # Save best model and print best loss
    torch.save(model, Path(f'./checkpoints/model_k{ga_k}_{ga_n_trial}.pth'))
    print('Best loss', loss)

    # Make some samples
    sample_from_decoder(model, image_shape=(3, *im_shape), means=means, stds=stds)