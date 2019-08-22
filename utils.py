from network import I3D, R3D, MDR21D, MDI3D, mobilenet_3D, slowfast, dnet
import torch
import torch.nn.functional as F
from network import I3D, R3D, MDR21D, MDI3D, mobilenet_3D, slowfast, dnet
from torch import nn

from configs import Config


def pairwise_similarity(x, y=None):
    if y is None:
        y = x
    # normalization
    y = normalize(y)
    x = normalize(x)
    # similarity
    similarity = torch.mm(x, y.t())
    return similarity


def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x


class SimLoss(nn.Module):
    def __init__(self, margin=0):
        super(SimLoss, self).__init__()
        self.margin = margin

    def forward(self, embed1, embed2):
        # Compute similarity matrix
        sim_mat = pairwise_similarity(embed1, embed2)
        loss = F.relu(sim_mat.diag().mean() - self.margin)
        return loss


# freeze bn
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


# unfreeze bn
def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def get_model_by_name(model_name='I3D', classes_num=101, pretrain=False, pretrain_path=None):
    if model_name == 'I3D':
        model = I3D.I3D_Classifier(num_classes=classes_num)
        if pretrain:
            model_state = model.i3d.state_dict()
            pretrained_state = torch.load(pretrain_path)
            pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_state}
            # model_state.update(pretrained_state)
            model.i3d.load_state_dict(pretrained_state)
    elif model_name == 'R3D':
        return R3D.resnet50(num_classes=classes_num)
        # model = R3D.resnet18(num_classes=classes_num,sample_size=112,sample_duration=32)
        # if pretrain:
        #     model_state = model.state_dict()
        #     pretrained_state = torch.load(pretrain_path)
        #     #pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_state}
        #     # model_state.update(pretrained_state)
        #     from collections import OrderedDict
        #     new_state_dict = OrderedDict()
        #     for k, v in pretrained_state['state_dict'].items():
        #         name = k[7:]  # remove module.
        #         new_state_dict[name] = v
        #     new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_state}
        #     new_state_dict = {k: v for k, v in new_state_dict.items() if 'fc' not in k}
        #     model.load_state_dict(new_state_dict,strict=False)
    elif model_name == 'MDR21D':
        return MDR21D.MDR2Plus1DClassifier(num_classes=classes_num, layer_sizes=[2, 2, 2, 2])
    elif model_name == 'MDI3D':
        model = MDI3D.MDI3D_Classifier(num_classes=classes_num)
    elif model_name == 'MMD3D':
        model = mobilenet_3D.MobileNetResidual(num_classes=classes_num)
    elif model_name == 'slowfast':
        model = slowfast.resnet50(class_num=classes_num)
    elif model_name == 'dnet50':
        model = dnet.dresnet50(class_num=classes_num)
    else:
        print(model_name + ' is not exist!')
        raise NotImplementedError
    return model


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def print_model_size(model):
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))


def print_config():
    config = Config()
    res = []
    for c in config.__dict__:
        res.append(c + ':' + str(config.__dict__[c]))
    config_str = '\n'.join(res)
    print(config_str)
    return config_str
