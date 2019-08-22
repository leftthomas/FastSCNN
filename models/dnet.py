import torch
import torch.nn.functional as F
from MDTConv.md_conv import DirectionalConv
from torch import nn

__all__ = ['resnet50', 'resnet101', 'resnet152', 'resnet200']


class MDConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super(MDConv, self).__init__()
        if not torch.cuda.is_available():
            raise EnvironmentError("only support for GPU mode")
        t_kernel_size = (3, 1, 1)
        t_stride = (stride, 1, 1)
        per_out_channels = out_channels // 5
        self.still = nn.Conv3d(in_channels, out_channels - 4 * per_out_channels, t_kernel_size, t_stride,
                               padding=(1, 0, 0), bias=bias)
        self.up = DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, None, bias=bias, mode='up')
        self.down = DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, None, bias=bias,
                                    mode='down')
        self.right = DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, None, bias=bias,
                                     mode='right')
        self.left = DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, None, bias=bias,
                                    mode='left')

    def forward(self, x):
        x1 = self.still(x)
        x2 = self.up(x)
        x3 = self.down(x)
        x4 = self.right(x)
        x5 = self.left(x)
        out = torch.cat([x1, x2, x3, x4, x5], 1)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            # self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),stride=(stride,1,1), bias=False, padding=(1, 0, 0))
            self.conv1 = MDConv(inplanes, planes, stride, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# class STBottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes,stride=1,mode = 's', head_conv=1,first_block = False):
#         super(STBottleneck, self).__init__()
#         self.first_block = first_block
#         if mode == 's':
#             res_stride = (1,stride,stride)
#             s_stride = (1,stride,stride)
#             t_stride = 1
#         else:
#             res_stride = (stride,stride,stride)
#             s_stride = (1,stride,stride)
#             t_stride = stride
#         if first_block:
#             self.res_conv = nn.Conv3d(inplanes,planes*4,kernel_size=1,stride=res_stride,padding=0,bias=False)
#             self.res_bn = nn.BatchNorm3d(planes*4)
#         if head_conv == 1:
#             self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1,stride=(t_stride,1,1), bias=False)
#             self.bn1 = nn.BatchNorm3d(planes)
#         elif head_conv == 3:
#             # if mode == 't':
#             # self.conv1 = MDConv(inplanes,planes,t_stride,bias=False)
#             # else:
#             self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),stride=(t_stride,1,1),bias=False, padding=(1, 0, 0))
#             self.bn1 = nn.BatchNorm3d(planes)
#         else:
#             raise ValueError("Unsupported head_conv!")
#         self.conv2 = nn.Conv3d(
#             planes, planes, kernel_size=(1, 3, 3), stride=s_stride, padding=(0, 1, 1), bias=False)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm3d(planes * 4)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.first_block:
#             residual = self.res_bn(self.res_conv(residual))
#         out += residual
#         out = self.relu(out)
#
#         return out

class SBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, head_conv=1, first_block=False):
        super(SBottleneck, self).__init__()
        self.first_block = first_block
        self.head_conv = head_conv
        res_stride = (1, stride, stride)
        s_stride = (1, stride, stride)
        t_stride = 1

        if first_block:
            self.res_conv = nn.Conv3d(inplanes, planes * 4, kernel_size=1, stride=res_stride, padding=0, bias=False)
            self.res_bn = nn.BatchNorm3d(planes * 4)
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=(t_stride, 1, 1), bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            # if mode == 't':
            # self.conv1 = MDConv(inplanes,planes,t_stride,bias=False)
            # else:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), stride=(t_stride, 1, 1), padding=(1, 0, 0),
                                   bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
            self.res_t_conv = nn.Conv3d(inplanes, planes, kernel_size=1, stride=(t_stride, 1, 1), padding=0, bias=False)
            self.res_t_bn = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=s_stride, padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.head_conv == 3:
            res_t = self.res_t_bn(self.res_t_conv(residual))
            out += res_t
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.first_block:
            residual = self.res_bn(self.res_conv(residual))
        out += residual
        out = self.relu(out)

        return out


class TBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, head_conv=1, first_block=False):
        super(TBottleneck, self).__init__()
        self.first_block = first_block
        res_stride = (stride, stride, stride)
        s_stride = (1, stride, stride)
        t_stride = stride
        if first_block:
            self.res_conv = nn.Conv3d(inplanes, planes * 4, kernel_size=1, stride=res_stride, padding=0, bias=False)
            self.res_bn = nn.BatchNorm3d(planes * 4)
            # self.conv1 = MDConv(inplanes,planes,t_stride,bias=False)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), stride=(t_stride, 1, 1), padding=(1, 0, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.res_t_conv = nn.Conv3d(inplanes, planes, kernel_size=1, stride=(t_stride, 1, 1), padding=0, bias=False)
        self.res_t_bn = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=s_stride, padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        res_t = self.res_t_bn(self.res_t_conv(residual))
        out += res_t
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.first_block:
            residual = self.res_bn(self.res_conv(residual))
        out += residual
        out = self.relu(out)

        return out


class STExchange(nn.Module):
    expansion = 4

    def __init__(self, s_in, t_in, n):
        super(STExchange, self).__init__()
        self.n = n
        # self.s2t = nn.Conv3d(s_in,t_in,1,1,padding=0,bias=False)
        self.t2s = nn.Conv3d(t_in, t_in, 1, (n, 1, 1), padding=0, bias=False)

    def forward(self, s, t):
        b, c, d, h, w = s.size()
        # s = self.s2t(s)
        if self.n != 1:
            s2t = F.interpolate(s, (d * self.n, h, w))
        else:
            s2t = s
        t2s = self.t2s(t)
        return torch.cat([t2s, s], 1), torch.cat([s2t, t], 1)


class STdownsample(nn.Module):
    expansion = 4

    def __init__(self, planes):
        super(STdownsample, self).__init__()
        self.t_1 = nn.Conv3d(4 * planes, 2 * planes, kernel_size=1, stride=(2, 1, 1), bias=False)
        self.s_1 = nn.Conv3d(4 * planes, 2 * planes, kernel_size=1, stride=(1, 2, 2), bias=False)
        self.t_bn1 = nn.BatchNorm3d(2 * planes)
        self.s_bn1 = nn.BatchNorm3d(2 * planes)
        self.relu = nn.ReLU()

    def forward(self, s, t):
        s = self.relu(self.s_bn1(self.s_1(s)))
        t = self.relu(self.t_bn1(self.t_1(t)))
        return torch.cat([s, t], 1)


class DNet(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3], class_num=10, dropout=0.5):
        super(DNet, self).__init__()

        self.s_inplanes = 32  # spatial = 32, temporal=32
        self.t_inplanes = 32
        self.expansion_num = [1, 4, 8, 16]
        self.t_i = 0
        self.s_i = 0
        self.s_planes = self.s_inplanes
        self.t_planes = self.t_inplanes
        self.s_f_planes = self.s_planes + self.t_planes
        self.t_f_planes = self.s_planes + self.t_planes

        self.s_conv1 = nn.Conv3d(3, self.s_inplanes, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                                 bias=False)
        self.s_bn1 = nn.BatchNorm3d(self.s_inplanes)
        self.s_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.t_input = nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=(0, 0, 0))
        self.t_conv1 = nn.Conv3d(3, self.t_inplanes, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1),
                                 bias=False)
        self.t_bn1 = nn.BatchNorm3d(self.t_inplanes)
        # self.t_conv2 = MDConv(32,32,2,bias=False)
        # self.t_conv2 = nn.Conv3d(self.t_inplanes, self.t_inplanes, kernel_size=(5, 1, 1), stride=(2, 1, 1), padding=(2, 0, 0), bias=False)
        # self.t_bn2 = nn.BatchNorm3d(self.t_inplanes)
        # self.t_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.relu = nn.ReLU(inplace=True)

        self.spatial_layer1 = self._make_layer(SBottleneck, self.s_inplanes, layers[0], mode='s', head_conv=1,
                                               first_layer=True)
        self.spatial_layer2 = self._make_layer(SBottleneck, self.s_inplanes * 2, layers[1], mode='s', head_conv=1)
        self.spatial_layer3 = self._make_layer(SBottleneck, self.s_inplanes * 4, layers[2], mode='s', head_conv=3)
        self.spatial_layer4 = self._make_layer(SBottleneck, self.s_inplanes * 8, layers[3], mode='s', head_conv=3)

        self.temporal_layer1 = self._make_layer(TBottleneck, self.t_inplanes, layers[0], mode='t', head_conv=3,
                                                first_layer=True)
        self.temporal_layer2 = self._make_layer(TBottleneck, self.t_inplanes * 2, layers[1], mode='t', head_conv=3)
        self.temporal_layer3 = self._make_layer(TBottleneck, self.t_inplanes * 4, layers[2], mode='t', head_conv=3)
        self.temporal_layer4 = self._make_layer(TBottleneck, self.t_inplanes * 8, layers[3], mode='t', head_conv=3)

        self.exchange0 = STExchange(self.s_inplanes, self.t_inplanes, 4)
        self.exchange1 = STExchange(self.s_inplanes * 4, self.t_inplanes * 4, 4)
        self.exchange2 = STExchange(self.s_inplanes * 8, self.t_inplanes * 8, 2)
        self.exchange3 = STExchange(self.s_inplanes * 16, self.t_inplanes * 16, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.s_planes + self.t_planes, class_num, bias=False)

    def forward(self, input):

        s = self.relu(self.s_bn1(self.s_conv1(input[:, :, ::8, :, :])))
        s = self.s_maxpool(s)
        t = self.t_input(input)
        t = self.relu(self.t_bn1(self.t_conv1(t)))
        # t = self.relu(self.t_bn2(self.t_conv2(t)))
        s, t = self.exchange0(s, t)
        s1 = self.spatial_layer1(s)
        t1 = self.temporal_layer1(t)

        s1, t1 = self.exchange1(s1, t1)
        s2 = self.spatial_layer2(s1)
        t2 = self.temporal_layer2(t1)

        s2, t2 = self.exchange2(s2, t2)
        s3 = self.spatial_layer3(s2)
        t3 = self.temporal_layer3(t2)

        s3, t3 = self.exchange3(s3, t3)
        s4 = self.spatial_layer4(s3)
        t4 = self.temporal_layer4(t3)

        s_out = nn.AdaptiveAvgPool3d(1)(s4)
        t_out = nn.AdaptiveAvgPool3d(1)(t4)
        out = torch.cat([s_out, t_out], 1)
        out = out.view(-1, out.size(1))
        out = self.drop(out)
        out = self.fc(out)
        return out

    def _make_layer(self, block, planes, blocks, mode='s', head_conv=3, first_layer=False):

        if first_layer:
            stride = 1
        else:
            stride = 2
        if mode == 's':
            layers = []
            layers.append(
                block(self.s_f_planes * self.expansion_num[self.s_i], planes, stride=stride, head_conv=head_conv,
                      first_block=True))
            self.s_i += 1
            self.s_planes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.s_planes, planes, stride=1, head_conv=head_conv))

        else:
            layers = []
            layers.append(
                block(self.t_f_planes * self.expansion_num[self.t_i], planes, stride=stride, head_conv=head_conv,
                      first_block=True))
            self.t_i += 1
            self.t_planes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.t_planes, planes, stride=1, head_conv=head_conv))
        return nn.Sequential(*layers)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = DNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = DNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = DNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = DNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def dresnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = DNet([3, 4, 6, 3], **kwargs)
    return model


if __name__ == "__main__":
    num_classes = 101
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 32, 112, 112).cuda())
    model = dresnet50(class_num=num_classes).cuda()
    output = model(input_tensor)
    print(model)
    print(output.size())
    # print(model)
    # print(model.bn1.weight)
