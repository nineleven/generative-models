from AE import Autoencoder

import numpy as np
import torch

def train_for_one_epoch(model, optimizer, loader, criterion, use_gpu=False):

    losses = []

    for x, y in loader:
        
        if use_gpu:
            x, y = x.cuda(), y.cuda()
        
        rec_x = model(x)
        
        loss = criterion(rec_x, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        
    mean_loss = np.mean(losses)
 
    return mean_loss

def train_autoencoder(n_latent_features, train_criterion, train_loader, 
                      image_shape, n_epochs=100, use_gpu=True,
                     num_conv_blocks=3, num_upsample_blocks=3):

    model = Autoencoder(n_latent_features, num_conv_blocks, num_upsample_blocks, image_shape)
    
    if use_gpu:
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    
    for epoch in range(n_epochs):
        mean_loss = train_for_one_epoch(model, optimizer, train_loader, train_criterion, use_gpu=use_gpu)
        print(f'epoch #{epoch} mean train loss: {mean_loss:.3f}')
            
    return model