import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

import numpy as np

class Encoder(nn.Module):
    
    def conv_block(self, in_filters, out_filters):
        return nn.Sequential(
            nn.Conv2d(in_filters, out_filters, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(out_filters),
        )
    
    def get_conv_block_list(self, n_latent_features, num_conv_blocks, in_filters=3, bottleneck=None):
        
        assert num_conv_blocks > 0
        
        if num_conv_blocks == 1 and bottleneck is not None:
            n_first_layer_filters = bottleneck
        else:
            n_first_layer_filters = 64
        
        conv_blocks_list = [self.conv_block(in_filters, n_first_layer_filters)]
        
        n_current_filters = n_first_layer_filters
        
        for _ in range(num_conv_blocks-2):
            current_block = self.conv_block(n_current_filters, n_current_filters * 2)
            conv_blocks_list.append(current_block)
            n_current_filters = n_current_filters * 2
            
        if num_conv_blocks > 1:
            if bottleneck is not None:
                current_block = self.conv_block(n_current_filters, bottleneck)
            else:
                current_block = self.conv_block(n_current_filters, n_current_filters * 2)
            conv_blocks_list.append(current_block)
            
        if bottleneck is not None:
            out_filters = bottleneck
        else:
            out_filters = n_current_filters * 2
            
        return conv_blocks_list, out_filters
        
    
    def __init__(self, n_latent_features, num_conv_blocks, in_filters, bottleneck=None):
        
        super(Encoder, self).__init__()
        
        conv_block_list, conv_out_fitlers = self.get_conv_block_list(n_latent_features, num_conv_blocks, in_filters, bottleneck)
        
        self.conv_blocks = nn.Sequential(*conv_block_list)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
            
        self.flattened_dim = conv_out_fitlers
        
        self.fc = nn.Linear(self.flattened_dim, n_latent_features)
        
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.avgpool(x)
        
        x = x.reshape(-1, self.flattened_dim)
        x = self.fc(x)
        
        return x
    
class Decoder(nn.Module):
    
    def upsample_block(self, in_filters, out_filters):
        return nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(in_filters, out_filters, 3, padding=1), 
            nn.ReLU(), 
            nn.BatchNorm2d(out_filters)
        )
    
    def get_upsample_block_list(self, num_upsample_blocks, in_filters, out_filters=3):
        assert num_upsample_blocks > 0

        conv_blocks_list = []
        
        n_current_filters = in_filters
        
        for _ in range(num_upsample_blocks-1):
            current_block = self.upsample_block(n_current_filters, n_current_filters // 2)
            conv_blocks_list.append(current_block)
            n_current_filters = n_current_filters // 2
            
        current_block = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(n_current_filters, out_filters, 3, padding=1)
        )
        conv_blocks_list.append(current_block)

        return conv_blocks_list
    
    def __init__(self, n_latent_features, num_upsample_blocks, output_shape, initial_filters=None):
        
        super(Decoder, self).__init__()
        
        assert all([dim >= 2**num_upsample_blocks for dim in output_shape[1:]])
        
        initial_shape = (dim // 2**num_upsample_blocks for dim in output_shape[1:])
        
        if initial_filters is None:
            initial_filters = 16 * 2 ** num_upsample_blocks
        
        self.reshape_dim = (initial_filters, *initial_shape)
        flattened_dim = int(np.product(self.reshape_dim))
        
        self.fc = nn.Sequential(
            nn.Linear(n_latent_features, flattened_dim), 
            nn.ReLU(), 
            nn.BatchNorm1d(flattened_dim)
        )
        
        conv_block_list = self.get_upsample_block_list(num_upsample_blocks, initial_filters, output_shape[0])
        
        self.conv_blocks = nn.Sequential(*conv_block_list)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc(x)
        
        x = x.reshape(-1, *self.reshape_dim)
        
        x = self.conv_blocks(x)

        x = self.sigmoid(x)
        
        return x
    
class Autoencoder(nn.Module):
    
    def __init__(self, n_latent_features, num_conv_blocks, num_upsample_blocks, image_shape, encoder_bottleneck=None):
        
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(n_latent_features, num_conv_blocks, image_shape[0], bottleneck=encoder_bottleneck)
        
        self.decoder = Decoder(n_latent_features, num_upsample_blocks, image_shape)
        
    def encode(x):
        return self.encoder(x)
        
    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)
        
        return x