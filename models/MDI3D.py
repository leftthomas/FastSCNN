import torch.nn as nn
from MDTConv.md_conv import MDConv


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)  # verify bias false

        # verify defalt value in sonnet
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MDCBR(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=(1, 0, 0)):
        super(MDCBR, self).__init__()
        self.conv = MDConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=False)  # verify bias false
        # verify defalt value in sonnet
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(192, 96, kernel_size=1, stride=1),
            # BasicConv3d(96, 128, kernel_size=3, stride=1, padding=1),
            MDCBR(96, 128, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),

        )
        self.branch2 = nn.Sequential(
            BasicConv3d(192, 16, kernel_size=1, stride=1),
            BasicConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
            # BasicConv3d(128, 192, kernel_size=3, stride=1, padding=1),
            MDCBR(128, 192, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(480, 96, kernel_size=1, stride=1),
            # BasicConv3d(96, 208, kernel_size=3, stride=1, padding=1),
            MDCBR(96, 208, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(480, 16, kernel_size=1, stride=1),
            BasicConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
            # BasicConv3d(112, 224, kernel_size=3, stride=1, padding=1),
            MDCBR(112, 224, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            BasicConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            # BasicConv3d(128, 256, kernel_size=3, stride=1, padding=1),
            MDCBR(128, 256, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            BasicConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            # BasicConv3d(144, 288, kernel_size=3, stride=1, padding=1),
            MDCBR(144, 288, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            # BasicConv3d(160, 320, kernel_size=3, stride=1, padding=1),
            MDCBR(160, 320, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 160, kernel_size=1, stride=1),
            # BasicConv3d(160, 320, kernel_size=3, stride=1, padding=1),
            MDCBR(160, 320, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            # BasicConv3d(192, 384, kernel_size=3, stride=1, padding=1),
            MDCBR(192, 384, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            BasicConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class MDI3D(nn.Module):
    def __init__(self):
        super(MDI3D, self).__init__()
        self.features = nn.Sequential(
            BasicConv3d(3, 64, kernel_size=7, stride=2, padding=3),  # (64, 32, 112, 112)
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # (64, 32, 56, 56)
            BasicConv3d(64, 64, kernel_size=1, stride=1),  # (64, 32, 56, 56)
            # BasicConv3d(64, 192, kernel_size=, stride=1, padding=1),  # (192, 32, 56, 56)
            MDCBR(64, 192, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # (192, 32, 28, 28)
            Mixed_3b(),  # (256, 32, 28, 28)
            Mixed_3c(),  # (480, 32, 28, 28)
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),  # (480, 16, 14, 14)
            Mixed_4b(),  # (512, 16, 14, 14)
            Mixed_4c(),  # (512, 16, 14, 14)
            Mixed_4d(),  # (512, 16, 14, 14)
            Mixed_4e(),  # (528, 16, 14, 14)
            Mixed_4f(),  # (832, 16, 14, 14)
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),  # (832, 8, 7, 7)
            Mixed_5b(),  # (832, 8, 7, 7)
            Mixed_5c(),  # (1024, 8, 7, 7)
        )

    def forward(self, x):
        x = self.features(x)
        return x


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        # self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, x):
        # x = self.linear(x)
        norm = x.norm(dim=1, p=2, keepdim=True)
        x = x.div(norm.expand_as(x))
        return x


class MDI3D_Classifier(nn.Module):
    def __init__(self, num_classes=101, dropout_drop_prob=0.5, spatial_squeeze=True, use_sim=False):
        super(MDI3D_Classifier, self).__init__()
        self.i3d = MDI3D()
        self.pool = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=1)  # (1024, 8, 1, 1)
        self.drop = nn.Dropout3d(dropout_drop_prob)
        self.classifier = nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True)  # (400, 8, 1, 1)
        # classifier = nn.Linear(1024, num_classes, bias=True)  # (400, 8, 1, 1)
        self.spatial_squeeze = spatial_squeeze
        self.__init_weight()

    def forward(self, x):
        x = self.i3d(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.classifier(x)
        if self.spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        averaged_logits = torch.mean(x, 2)
        # print(averaged_logits.size())
        # predictions = self.softmax(averaged_logits)
        return averaged_logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    import torch

    net = MDI3D_Classifier().cuda()
    # model_state = net.i3d.state_dict()
    # pretrained_state = torch.load('rgb_imagenet.pkl')
    # pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_state}
    # model_state.update(pretrained_state)
    inputs = torch.rand(4, 3, 64, 112, 112).cuda()
    # net.i3d.load_state_dict(pretrained_state)
    # torch.save(net, 'params.pth.tar')
    # print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    outputs = net.forward(inputs)
    print(outputs.size())
    # print(pairwise_similarity(embed))
    # net = R2Plus1DClassifier(101, (3, 4, 6, 3), pretrained=False)
    # print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    # outputs = net.forward(inputs)
    # print(outputs.size())
