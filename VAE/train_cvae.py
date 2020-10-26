import torch
import torch.nn as nn

from model_cvae import ConditionalVAE

import matplotlib.pyplot as plt


def initialize_model(n_latent_features, n_classes, use_gpu=True):
    model = ConditionalVAE(n_latent_features, n_classes)
    
    if use_gpu:
        model = model.cuda()
        
    return model

def make_vae_criterion(rec_criterion, KLD_N):
    
    def KLD(mu, logvar):        
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld /= KLD_N
        
        return kld
    
    def criterion(model_output, x):
        x_rec, mu, logvar = model_output
        
        return KLD(mu, logvar) + rec_criterion(x_rec, x)
    
    return criterion

def train_model(model, rec_criterion, KLD_N, optimizer, loader, n_epochs):
    
    criterion = make_vae_criterion(rec_criterion, KLD_N)
    
    model.train()
    
    for epoch in range(n_epochs):
        mean_loss = train_epoch(model, optimizer, loader, criterion)
        print(f'epoch #{epoch} mean train loss: {mean_loss:.3f}')

def train_epoch(model, optimizer, loader, criterion):

    losses = []

    for x, y in loader:
        y = y.squeeze()
        
        model_output = model(x, y)
        
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
       
    y = y.squeeze()
        
    rec_x = model.decode(model.encode(x, y), y)

    x_numpy = x.cpu().detach().numpy()
    rec_x_numpy = rec_x.cpu().detach().numpy()
    y_numpy = y.cpu().detach().numpy()
    
    nrows, ncols = len(y_numpy), 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

    for i in range(len(y_numpy)):
        channel_last_x = np.transpose(x_numpy[i], (1, 2, 0))
        class_number = np.argmax(y_numpy[i])
        
        ax[i][0].imshow(channel_last_x)
        ax[i][0].set_title(f'actual {classes[class_number]}')
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        
        channel_last_x_rec = np.transpose(rec_x_numpy[i], (1, 2, 0))
        
        ax[i][1].imshow(channel_last_x_rec)
        ax[i][1].set_title('reconstructed image')
        ax[i][1].set_xticks([])
        ax[i][1].set_yticks([])