import torch
import torch.nn as nn


class Encoder(nn.Module):
    
    def __init__(self, n_latent_features):
        
        super(Encoder, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=2), 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1, stride=2), 
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32*8*8, n_latent_features), nn.ReLU(), nn.BatchNorm1d(n_latent_features),
            nn.Linear(n_latent_features, n_latent_features)
        )
        
    def forward(self, x):
        x = self.conv(x)
        
        x = x.reshape(-1, 32*8*8)
        
        x = self.fc(x)
        
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, n_latent_features):
        
        super(Decoder, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(n_latent_features, 32*8*8), nn.ReLU()
        )
    
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.fc(x)
        
        x = x.reshape(-1, 32, 8, 8)
        
        x = self.conv(x)
        
        return x
    
class Autoencoder(nn.Module):
    
    def __init__(self, n_latent_features):
        
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(n_latent_features)
        
        self.decoder = Decoder(n_latent_features)
        
    def encode(self, x):
        return self.encoder(x)

    
    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)
        
        return x