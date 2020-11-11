import torch
import torch.nn as nn

class Generator(nn.Module):
    
    def __init__(self, n_latent_features):
        
        super(Generator, self).__init__()
        
        self.__n_latent_features = n_latent_features
        
        self.fc = nn.Sequential(
            nn.Linear(n_latent_features, 64*8*8), nn.LeakyReLU()
        )
        
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
    @property
    def n_latent_features(self):
        return self.__n_latent_features
        
    def forward(self, z):
        x = self.fc(z)

        x = x.reshape(-1, 64, 8, 8)

        x = self.conv(x)
        
        return x
    
class Discriminator(nn.Module):
    
    def __init__(self):
        
        super(Discriminator, self).__init__()
        
        self.disc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), 
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool2d(1),
            
            nn.Flatten(),
            
            nn.Linear(64, 1), nn.Sigmoid()
        )
        
        
    def forward(self, x):
        x = self.disc(x)

        return x
    
class GAN(nn.Module):
    
    def __init__(self, n_latent_features):
        
        super(GAN, self).__init__()
        
        self.__generator = Generator(n_latent_features)
        self.__discriminator = Discriminator()
        
    @property
    def generator(self):
        return self.__generator
    
    @property
    def discriminator(self):
        return self.__discriminator