import torch
import torch.nn as nn

from model_gan import GAN

import numpy as np


def initialize_model(n_latent_features, device):
    return GAN(n_latent_features).to(device)

def generate_examples(generator, num_examples, device):
    z = torch.randn(num_examples, generator.n_latent_features, device=device)
    return generator(z).cpu().detach().numpy()

def train_model(model, criterion, loader, n_epochs, device, *, 
                generator_optimizer, discriminator_optimizer):
    
    d_steps = 1
    g_steps = 1
    
    examples = []
    
    for epoch in range(n_epochs):
        
        mean_d_loss, mean_g_loss = train_epoch(model, criterion, loader, g_steps, d_steps, device,
                                               generator_optimizer=generator_optimizer,
                                               discriminator_optimizer=discriminator_optimizer)

        print(f'epoch #{epoch} mean discriminator loss: {mean_d_loss:.3g}, mean generator loss: {mean_g_loss:.3g}')
        
        epoch_examples = generate_examples(model.generator, 4, device)
        examples.append(epoch_examples)
        
    return examples
        
def train_epoch(model, criterion, loader, g_steps, d_steps, device, *,
                generator_optimizer, discriminator_optimizer,):

    d_epoch_losses = []
    g_epoch_losses = []
    
    batches = []
    
    switch_steps = 3
    
    training_generator = False
    num_steps = 0
    
    iterator = loader.__iter__()

    while True:
        
        if training_generator:
            g_loss, g_acc = generator_step(model.generator, model.discriminator, 
                                        generator_optimizer, len(batch[0]), criterion, device)
            g_epoch_losses.append(g_loss)
        else:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            d_loss, d_acc = discriminator_step(model.generator, model.discriminator, 
                                            discriminator_optimizer, batch, criterion, device)
            d_epoch_losses.append(d_loss)
            
        num_steps += 1
        
        if num_steps == switch_steps:
            training_generator = not training_generator
            num_steps = 0
            
    return np.mean(d_epoch_losses), np.mean(g_epoch_losses)
        

def generator_step(generator, discriminator, optimizer, batch_size, criterion, device):
    
    z = torch.randn(batch_size, generator.n_latent_features).to(device)
    
    fake_images = generator(z)
    y_pred = discriminator(fake_images)
    
    misleading_labels = torch.zeros(batch_size, 1).to(device)
        
    loss = criterion(y_pred, misleading_labels)
    accuracy = torch.mean((y_pred.round() == misleading_labels).float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), accuracy.item()

def discriminator_step(generator, discriminator, optimizer, batch, criterion, device):
    real_images = batch[0]
    
    z = torch.randn(len(real_images), generator.n_latent_features).to(device)
    fake_images = generator(z).detach()
    
    images = torch.cat([real_images, fake_images], dim=0)
    labels = torch.cat([
        torch.zeros(len(real_images)),
        torch.ones(len(fake_images)),
    ], dim=0).to(device)

    y_pred = discriminator(images).squeeze()

    loss = criterion(y_pred, labels)
    accuracy = torch.mean((y_pred.round() == labels).float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), accuracy.item()