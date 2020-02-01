import torch

from model import FastSCNN

if __name__ == '__main__':
    model = FastSCNN(in_channels=3, num_classes=10)
    x = torch.rand(2, 3, 256, 256)
    y = model(x)
