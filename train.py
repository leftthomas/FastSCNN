import argparse

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Cityscapes
from model import FastSCNN
from utils import PolynomialLRScheduler


# train for one epoch
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data, target in train_bar:
        data, target = data.to('cuda'), target.to('cuda')
        train_optimizer.zero_grad()
        out = net(data)
        loss = loss_criterion(out, target)
        loss.backward()
        train_optimizer.step()

        total_num += data.size(0)
        total_loss += loss.item() * data.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# # test for one epoch
# def test(net, test_data_loader):
#     net.eval()
#     total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
#     with torch.no_grad():
#         # loop test data to predict the label by weighted knn search
#         test_bar = tqdm(test_data_loader)
#         for data, target, _ in test_bar:
#             data, target = data.to('cuda'), target.to('cuda')
#             output = net(data)
#
#             total_num += data.size(0)
#             pred_labels = pred_scores.argsort(dim=-1, descending=True)
#             total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
#             total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
#             test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
#                                      .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))
#
#     return total_top1 / total_num * 100, total_top5 / total_num * 100


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

    # dataset, model setup, optimizer config and loss definition
    train_data = Cityscapes(root=data_path, split='train', crop_size=(crop_h, crop_w))
    val_data = Cityscapes(root=data_path, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    model = FastSCNN(in_channels=3, num_classes=19).to('cuda')
    optimizer = optim.SGD(model.parameters(), lr=0.045, momentum=0.9)
    print("# trainable model parameters:", sum(param.numel() if param.requires_grad else 0
                                               for param in model.parameters()))
    lr_scheduler = PolynomialLRScheduler(optimizer, max_decay_steps=epochs, power=0.9)
    loss_criterion = nn.CrossEntropyLoss(ignore_index=255)

    # training loop
    results = {'train_loss': [], 'test_loss': [], 'train_mIOU': [], 'test_mIOU': []}
    best_miou = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        # test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        # results['test_acc@1'].append(test_acc_1)
        # results['test_acc@5'].append(test_acc_5)
        # save statistics
        # data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        # data_frame.to_csv('results/{}_{}_results.csv'.format(crop_h, crop_w), index_label='epoch')
        lr_scheduler.step()
        # if test_acc_1 > best_miou:
        #     best_miou = test_acc_1
        #     torch.save(model.state_dict(), 'results/{}_{}_model.pth'.format(crop_h, crop_w))
