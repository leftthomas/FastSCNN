import os
import timeit
from datetime import datetime

import horovod.torch as hvd
from dataloaders.dataset import VideoDataset
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import *
from warmup import GradualWarmupScheduler

torch.backends.cudnn.benchmark = True

# init horovod and set backends
hvd.init()
torch.cuda.set_device(hvd.local_rank())

config = Config()

# set log and save dir,config

date = datetime.now().strftime('%m-%d-%H')
model_dir = os.path.join('models', config.model_name)
save_dir = os.path.join(model_dir, date)
save_mode_lame = config.model_name + '-' + config.dataset

if hvd.rank() == 0:
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_config = print_config()
    with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
        f.write(save_config)
    writer = SummaryWriter(log_dir=save_dir)


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def train_model(dataset=config.dataset, save_dir=save_dir, lr=config.lr, num_epochs=config.epoch_num,
                save_epoch=config.save_freq, useTest=config.use_test, test_interval=config.test_freq):
    model = get_model_by_name(config.model_name, config.classes_num, config.pretrain, config.pretrain_path)
    if hvd.rank() == 0:
        print_model_size(model)

    model = model.cuda()
    # model = model.to(config.device_ids[0])
    # if len(config.device_ids) > 1:
    #     if torch.cuda.device_count() >= len(config.device_ids):
    #         model = nn.DataParallel(model, device_ids=config.device_ids)
    #     else:
    #         raise ValueError("the machine don't have {} gpus".format(str(len(config.device_ids))))

    criterion = nn.CrossEntropyLoss()
    if config.set_optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    elif config.set_optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr / config.multiplier, momentum=0.9,
                              weight_decay=config.weight_decay)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    # scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epoch_num - config.warmup_num)
    #
    scheduler_mtstep = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=config.multiplier, total_epoch=config.warmup_num,
                                       after_scheduler=scheduler_mtstep)

    # resume
    if config.resume_epoch_num != 0:
        checkpoint = torch.load(config.resume_model_path)
        model.load_state_dict(checkpoint['state_dict'])
    if hvd.rank() == 0:
        print('Training model on {} dataset...'.format(dataset))
    train_dataset = VideoDataset(dataset=dataset, split='train', clip_len=config.clip_len)
    test_dataset = VideoDataset(dataset=dataset, split='test', clip_len=config.clip_len)
    train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_sampler = DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, sampler=test_sampler)
    train_size = len(train_sampler)
    test_size = len(test_sampler)

    for epoch in range(config.resume_epoch_num, num_epochs):
        # each epoch has a training and validation step
        start_time = timeit.default_timer()

        # reset the running loss and corrects
        train_loss = 0.0
        # loss2 = 0.0
        train_accuracy = 0.0

        # set model to train() or eval() mode depending on whether it is trained
        # or being validated. Primarily affects layers such as BatchNorm or Dropout.
        scheduler.step(epoch)
        print("current lr is :", optimizer.param_groups[0]['lr'])
        model.train()
        if config.freeze_bn:
            model.apply(set_bn_eval)

        i = 0
        for inputs, labels in tqdm(train_dataloader):
            # move inputs and labels to the device the training is taking place on
            # inputs = inputs.to(config.device_ids[0])
            # labels = labels.to(config.device_ids[0])
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            i += 1
            outputs = model(inputs)
            # shuffled_outputs, shuffled_embed = model(shuffled_inputs)
            # simloss = criterion2(embed, shuffled_embed)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            # loss = criterion(outputs, labels) + 0.2 * simloss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            # loss2 += simloss.item() * inputs.size(0)
            if i % 30 == 0:
                # print('training loss %f,sim loss %f' % (running_loss / i / config.batch_size, loss2 / i / config.batch_size))
                print('training loss %f' % (train_loss / i / config.batch_size))
            train_accuracy += torch.sum(preds == labels.data)

        train_loss = train_loss / train_size
        train_accuracy = train_accuracy.double() / train_size

        train_loss = metric_average(train_loss, 'avg_loss')
        train_accuracy = metric_average(train_accuracy, 'avg_accuracy')
        if hvd.rank() == 0:
            writer.add_scalar('data/train_loss_epoch', train_loss, epoch)
            writer.add_scalar('data/train_acc_epoch', train_accuracy, epoch)
            print("[train] Epoch: {}/{} Loss: {} Acc: {}".format(epoch, config.epoch_num, train_loss, train_accuracy))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1) and hvd.rank() == 0:
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(),
                        }, os.path.join(save_dir, save_mode_lame + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, save_mode_lame + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            test_loss = 0.0
            test_accuracy = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                test_accuracy += torch.sum(preds == labels.data)

            test_loss = test_loss / test_size
            test_accuracy = test_accuracy.double() / test_size

            # all reduce
            test_loss = metric_average(test_loss, 'avg_loss')
            test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

            if hvd.rank() == 0:
                writer.add_scalar('data/test_loss_epoch', test_loss, epoch)
                writer.add_scalar('data/test_acc_epoch', test_accuracy, epoch)

                print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch, config.epoch_num, test_loss, test_accuracy))
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")
    if hvd.rank() == 0:
        writer.close()


if __name__ == "__main__":
    train_model()
