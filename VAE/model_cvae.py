import torch
import torch.nn as nn


class ConditionalEncoder(nn.Module):
    
    def __init__(self, n_latent_features, n_classes):
        
        super(ConditionalEncoder, self).__init__()
        
        self.n_classes = n_classes
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), 
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32 + n_classes, 32, 3, padding=1, stride=2), 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1,), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64*8*8, n_latent_features), nn.ReLU(), nn.BatchNorm1d(n_latent_features)
        )
        
        self.fc_mu = nn.Linear(n_latent_features, n_latent_features)
        self.fc_logvar = nn.Linear(n_latent_features, n_latent_features)
        
    def append_class_labels(self, x, y):
        batch_size = y.shape[0]
        y = y.repeat(1, x.shape[2] * x.shape[3]).flatten()
        y = y.view(batch_size, x.shape[2], x.shape[3], self.n_classes)
        y = y.permute(0, 3, 1, 2)
        
        return torch.cat([x, y], dim=1)
        
    def forward(self, x, y):
        x = self.conv1(x)
        
        x = self.append_class_labels(x, y)
        
        x = self.conv2(x)
        
        x = x.reshape(-1, 64*8*8)
        
        x = self.fc(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
class ConditionalDecoder(nn.Module):
    
    def __init__(self, n_latent_features, n_classes):
        
        super(ConditionalDecoder, self).__init__()
        
        self.n_classes = n_classes
        
        self.fc = nn.Sequential(
            nn.Linear(n_latent_features + n_classes, 64*8*8), nn.ReLU()
        )
    
        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        
        x = self.fc(x)
        
        x = x.reshape(-1, 64, 8, 8)
        
        x = self.conv(x)
        
        return x
    
class ConditionalVAE(nn.Module):
    
    def __init__(self, n_latent_features, n_classes):
        
        super(ConditionalVAE, self).__init__()
        
        self.encoder = ConditionalEncoder(n_latent_features, n_classes)
        
        self.decoder = ConditionalDecoder(n_latent_features, n_classes)
        
    def encode(self, x, y):
        mu, logvar = self.encoder(x, y)
        return mu
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def decode(self, x, y):
        return self.decoder(x, y)
        
    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        
        x = self.reparameterize(mu, logvar)

        x = self.decoder(x, y)
        
        return x, mu, logvar