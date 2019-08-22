import math

import torch.nn as nn
from MDTConv.md_conv import MDConv


class DepthWiseBlock(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super(DepthWiseBlock, self).__init__()
        # self.conv1 = nn.Conv3d(
        #     inp,
        #     inp,
        #     kernel_size=3,
        #     stride=stride,
        #     padding=1,
        #     groups=inp,
        #     bias=False)
        self.conv1 = MDConv(inp, inp, (3, 1, 1), stride, (1, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(inp)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            inp,
            oup,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn2 = nn.BatchNorm3d(oup)
        self.inplanes = inp
        self.outplanes = oup
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        out = self.relu(out)

        return out


class MobileNetResidual(nn.Module):
    @staticmethod
    def conv_bn(inp, oup, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv3d(
                inp,
                oup,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False),
            nn.BatchNorm3d(oup),
            nn.ReLU(inplace=True)
        )

    def __init__(self, num_classes=400, last_fc=True):
        super(MobileNetResidual, self).__init__()
        self.__init_weight()
        self.last_fc = last_fc

        self.model = nn.Sequential(
            self.conv_bn(3, 32, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
            DepthWiseBlock(32, 64, 1),
            DepthWiseBlock(64, 128, (1, 2, 2)),
            DepthWiseBlock(128, 128, 1),
            DepthWiseBlock(128, 256, 2),
            DepthWiseBlock(256, 256, 1),
            DepthWiseBlock(256, 512, 2),
            DepthWiseBlock(512, 512, 1),
            DepthWiseBlock(512, 512, 1),
            DepthWiseBlock(512, 512, 1),
            DepthWiseBlock(512, 512, 1),
            DepthWiseBlock(512, 512, 1),
            DepthWiseBlock(512, 1024, 2),
            DepthWiseBlock(1024, 1024, 1),
        )

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.5)

        self.linear = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.model(x)

        x = self.avgpool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        if self.last_fc:
            x = self.linear(x)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for the conv layer of the net.
    """
    b = [model.model]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the fc layer of the net.
    """
    b = [model.linear]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    import torch

    inputs = torch.rand(4, 3, 16, 112, 112).cuda()
    net = MobileNetResidual(112, 16, 101, True).cuda()
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    outputs = net.forward(inputs)
    print(outputs.size())
