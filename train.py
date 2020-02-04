import argparse
import os
import time

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Cityscapes
from model import FastSCNN
from utils import PolynomialLRScheduler

os.environ['CITYSCAPES_DATASET'] = 'results/'


# train or val for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct, total_time, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(), target.cuda()
            torch.cuda.synchronize()
            start_time = time.time()
            out = net(data)
            prediction = torch.argmax(out, dim=1)
            torch.cuda.synchronize()
            end_time = time.time()
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_time += end_time - start_time
            total_loss += loss.item() * data.size(0)
            total_correct += torch.sum(prediction == target).item() / target.numel() * data.size(0)

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} mPA: {:.2f}% FPS: {:.0f}'
                                     .format('Train' if is_train else 'Val', epoch, epochs, total_loss / total_num,
                                             total_correct / total_num * 100, total_num / total_time))

    return total_loss / total_num, total_correct / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Fast-SCNN')
    parser.add_argument('--data_path', default='/home/data/cityscapes', type=str,
                        help='Data path for cityscapes dataset')
    parser.add_argument('--crop_h', default=1024, type=int, help='Crop height for training images')
    parser.add_argument('--crop_w', default=2048, type=int, help='Crop width for training images')
    parser.add_argument('--batch_size', default=12, type=int, help='Number of data for each batch to train')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    data_path, crop_h, crop_w = args.data_path, args.crop_h, args.crop_w
    batch_size, epochs = args.batch_size, args.epochs

    # dataset, model setup and optimizer config
    train_data = Cityscapes(root=data_path, split='train', crop_size=(crop_h, crop_w))
    val_data = Cityscapes(root=data_path, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    model = FastSCNN(in_channels=3, num_classes=19).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.045, momentum=0.9, weight_decay=4e-5)

    # model profile, learning scheduler and loss definition
    input = torch.randn(1, 3, crop_h, crop_w).cuda()
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    del input
    lr_scheduler = PolynomialLRScheduler(optimizer, max_decay_steps=epochs, power=0.9)
    loss_criterion = nn.CrossEntropyLoss(ignore_index=255)

    # training loop
    results = {'train_loss': [], 'val_loss': [], 'train_mPA': [], 'val_mPA': []}
    best_mPA = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_mPA = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_mPA'].append(train_mPA)
        val_loss, val_mPA = train_val(model, val_loader, None)
        results['val_loss'].append(val_loss)
        results['val_mPA'].append(val_mPA)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_{}_statistics.csv'.format(crop_h, crop_w), index_label='epoch')
        lr_scheduler.step()
        if val_mPA > best_mPA:
            best_mPA = val_mPA
            torch.save(model.state_dict(), 'results/{}_{}_model.pth'.format(crop_h, crop_w))
