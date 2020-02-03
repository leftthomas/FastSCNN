import torch
from torch.utils.data import DataLoader

from dataset import Cityscapes
from model import FastSCNN

if __name__ == '__main__':
    train_data = Cityscapes(root='/home/data/cityscapes', split='train', crop_size=(1024, 2048))
    val_data = Cityscapes(root='/home/data/cityscapes', split='val')
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=8)
    a = next(iter(train_loader))
    b = next(iter(val_loader))
    model = FastSCNN(in_channels=3, num_classes=10)
    x = torch.rand(2, 3, 256, 256)
    y = model(x)
