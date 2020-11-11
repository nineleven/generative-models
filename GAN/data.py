import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose

def to_gpu(data):
    if isinstance(data, (list, tuple)):
        return [to_gpu(item) for item in data]
    return data.cuda()

class GPULoader:
    
    def __init__(self, loader):
        self.loader = loader
        
    def __iter__(self):
        for batch in self.loader:
            yield to_gpu(batch)
                
    def __len__(self):
        return len(self.loader)
    
def load_data(root, train_transform, test_transform, target_transform, batch_size, use_gpu):

    train_dataset = torchvision.datasets.CIFAR10(root=root, train=True,
                                           download=True, transform=train_transform, target_transform=target_transform)

    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=test_transform, target_transform=target_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    if use_gpu:
        train_loader = GPULoader(train_loader)
        test_loader = GPULoader(test_loader)
    
    return train_loader, test_loader

class OneHot:
    def __init__(self, n_classes):
        self.n_classes = n_classes
    
    def __call__(self, y):
        return nn.functional.one_hot(torch.tensor([y]), self.n_classes)

def combine_transforms(*, preprocessing=[], augmentations=[], postprocessing=[], target=[]):

    train_transform = Compose(preprocessing + augmentations + postprocessing)
    test_transform = Compose(preprocessing + postprocessing)
    
    target_transform = Compose(target)

    return train_transform, test_transform, target_transform