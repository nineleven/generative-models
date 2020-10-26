import torch
import torch.nn as nn
from torchvision.transforms import Compose

from model import Autoencoder

import numpy as np

import matplotlib.pyplot as plt


def initialize_model(n_latent_features, use_gpu=True):
    model = Autoencoder(n_latent_features)
    
    if use_gpu:
        model = model.cuda()
        
    return model


def train_model(model, rec_criterion, optimizer, loader, n_epochs):

    model.train()
    
    for epoch in range(n_epochs):
        mean_loss = train_epoch(model, optimizer, loader, rec_criterion)
        print(f'epoch #{epoch} mean train loss: {mean_loss:.3f}')

def train_epoch(model, optimizer, loader, criterion):

    losses = []

    for x, y in loader:
        model_output = model(x)
        
        loss = criterion(model_output, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
    mean_loss = np.mean(losses)
    
    return mean_loss


def evaluate_model(model, loader, max_images=20):
    
    classes = np.array(('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck'))
    
    x, y = None, None
    for batch in loader:
        x, y = batch[:max_images]
        
    rec_x = model.decode(model.encode(x))

    x_numpy = x.cpu().detach().numpy()
    rec_x_numpy = rec_x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    
    nrows, ncols = len(y), 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

    for i in range(len(y)):
        channel_last_x = np.transpose(x_numpy[i], (1, 2, 0))
        ax[i][0].imshow(channel_last_x)
        ax[i][0].set_title(f'actual {classes[y[i]]}')
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        
        channel_last_x_rec = np.transpose(rec_x_numpy[i], (1, 2, 0))
        ax[i][1].imshow(channel_last_x_rec)
        ax[i][1].set_title('reconstructed image')
        ax[i][1].set_xticks([])
        ax[i][1].set_yticks([])