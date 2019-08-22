import torch.nn as nn
from torch.nn.modules.utils import _triple


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, first_conv=False,
                 down_sample=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        self.down_sample = down_sample
        if first_conv:
            # decomposing the parameters into spatial and temporal components by
            # masking out the values with the defaults on the axis that
            # won't be convolved over. This is necessary to avoid unintentional
            # behavior such as padding being added twice
            spatial_kernel_size = kernel_size
            spatial_stride = (1, stride[1], stride[2])
            spatial_padding = padding

            temporal_kernel_size = (3, 1, 1)
            temporal_stride = (stride[0], 1, 1)
            temporal_padding = (1, 0, 0)

            # the spatial conv is effectively a 2D conv due to the
            # spatial_kernel_size, followed by batch_norm and ReLU
            self.spatial_conv = nn.Conv3d(in_channels, out_channels, spatial_kernel_size,
                                          stride=spatial_stride, padding=spatial_padding, bias=bias)
            self.bn1 = nn.BatchNorm3d(out_channels)
            # the temporal conv is effectively a 1D conv, but has batch norm
            # and ReLU added inside the model constructor, not here. This is an
            # intentional design choice, to allow this module to externally act
            # identical to a standard Conv3D, so it can be reused easily in any
            # other codebase
            # self.temporal_conv = MDConv(out_channels, out_channels, temporal_kernel_size, temporal_stride,temporal_padding, bias=bias)
            self.temporal_conv = nn.Conv3d(out_channels, out_channels, temporal_kernel_size,
                                           stride=temporal_stride, padding=temporal_padding, bias=bias)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU()
        else:
            # decomposing the parameters into spatial and temporal components by
            # masking out the values with the defaults on the axis that
            # won't be convolved over. This is necessary to avoid unintentional
            # behavior such as padding being added twice
            spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
            spatial_stride = (1, stride[1], stride[2])
            spatial_padding = (0, padding[1], padding[2])

            temporal_kernel_size = (kernel_size[0], 1, 1)
            temporal_stride = (1, 1, 1)
            temporal_padding = (padding[0], 0, 0)

            # compute the number of intermediary channels (M) using formula
            # from the paper section 3.5

            # the spatial conv is effectively a 2D conv due to the
            # spatial_kernel_size, followed by batch_norm and ReLU
            self.spatial_conv = nn.Conv3d(in_channels, out_channels, spatial_kernel_size,
                                          stride=spatial_stride, padding=spatial_padding, bias=bias)
            self.bn1 = nn.BatchNorm3d(out_channels)

            # the temporal conv is effectively a 1D conv, but has batch norm
            # and ReLU added inside the model constructor, not here. This is an
            # intentional design choice, to allow this module to externally act
            # identical to a standard Conv3D, so it can be reused easily in any
            # other codebase

            # self.temporal_conv = MDConv(out_channels, out_channels, temporal_kernel_size, temporal_stride,temporal_padding, bias=bias)
            self.temporal_conv = nn.Conv3d(out_channels, out_channels, temporal_kernel_size,
                                           stride=temporal_stride, padding=temporal_padding, bias=bias)
            if self.down_sample:
                self.tp = nn.MaxPool3d((2, 1, 1), (2, 1, 1))
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.spatial_conv(x)))
        x = self.relu(self.bn2(self.temporal_conv(x)))
        if self.down_sample:
            x = self.tp(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.conv1x1x1 = nn.Conv3d(in_channels, out_channels, 1, 2, 0)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
            self.pool = nn.MaxPool3d(2, 2)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2,
                                            down_sample=self.downsample)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        if self.downsample:
            x = self.downsamplebn(self.conv1x1x1(x))
        return self.relu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class MDR2Plus1DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(MDR2Plus1DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 1x7x7
        self.conv1 = SpatioTemporalConv(3, 60, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), first_conv=True)
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(60, 60, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(60, 120, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(120, 240, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(240, 480, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)

        return x.view(-1, 480)


class MDR2Plus1DClassifier(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, num_classes, layer_sizes=[2, 2, 2, 2], block_type=SpatioTemporalResBlock, pretrained=False):
        super(MDR2Plus1DClassifier, self).__init__()

        self.res2plus1d = MDR2Plus1DNet(layer_sizes, block_type)
        self.linear = nn.Linear(480, num_classes)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.res2plus1d(x)
        logits = self.linear(x)

        return logits

    def __load_pretrained_weights(self):
        s_dict = self.state_dict()
        for name in s_dict:
            print(name)
            print(s_dict[name].size())

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
    b = [model.res2plus1d]
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
    net = MDR2Plus1DClassifier(101, (3, 4, 6, 3), pretrained=False).cuda()
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    outputs = net.forward(inputs)
    print(outputs.size())
