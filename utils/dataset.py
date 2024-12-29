from torchvision import datasets
import numpy as np
from PIL import Image

class CIFAR10Dataset:
    def __init__(self, root="./data", train=True, transform=None):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        
        # Calculate mean and std
        if train:
            data = np.array(self.dataset.data, dtype=np.float32)
            self.mean = data.mean(axis=(0,1,2))/255.0
            self.std = data.std(axis=(0,1,2))/255.0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Convert PIL Image to numpy array
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        if self.transform:
            img = self.transform(img, train=self.dataset.train)
        return img, label 