#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import mdconv
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import init
from torch.nn.modules.utils import _triple


class MDConvFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, kernel_size, stride, padding, cof):
        ctx.stride = stride
        ctx.padding = padding
        ctx.kernel_size = kernel_size
        ctx.cof = cof
        output = mdconv.forward(input, weight, bias,
                                ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
                                ctx.stride[0], ctx.stride[1], ctx.stride[2],
                                ctx.padding[0], ctx.padding[1], ctx.padding[2],
                                cof[0], cof[1], cof[2], cof[3])
        ctx.save_for_backward(input, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = mdconv.backward(input, weight, bias, grad_output,
                                                             ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
                                                             ctx.stride[0], ctx.stride[1], ctx.stride[2],
                                                             ctx.padding[0], ctx.padding[1], ctx.padding[2],
                                                             ctx.cof[0], ctx.cof[1], ctx.cof[2], ctx.cof[3])

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class DirectionalConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding=None, bias=False, mode='down'):
        super(DirectionalConv, self).__init__()

        if not torch.cuda.is_available():
            raise EnvironmentError("only support for GPU mode")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.use_bias = bias

        if mode == 'up':
            self.cof = [-1, kernel_size[0] - 1, 0, 0]
            self.padding = (kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[2] // 2)
        elif mode == 'down':
            self.cof = [1, 0, 0, 0]
            self.padding = (kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[2] // 2)
        elif mode == 'right':
            self.cof = [0, 0, 1, 0]
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[0] // 2)
        elif mode == 'left':
            self.cof = [0, 0, -1, kernel_size[0] - 1]
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[0] // 2)
        else:
            raise ValueError("no such mode")

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return MDConvFunction.apply(input, self.weight, self.bias,
                                    self.kernel_size, self.stride, self.padding, self.cof)


# class MDConv(nn.Module):
#     def __init__(self, in_channels, out_channels,
#                  kernel_size=3, stride=1, padding=1 ,  dilation=1, groups=1, bias=False,ratial = 0.2,t_downsample = False):
#         super(MDConv, self).__init__()
#         if not torch.cuda.is_available():
#             raise EnvironmentError("only support for GPU mode")
#         per_out_channels = int((1-ratial)/4*out_channels)
#         t_kernel_size = (3, 1 , 1 )
#         s_kernel_size = (1,kernel_size,kernel_size)
#
#         if t_downsample:
#             t_stride = (stride,1,1)
#         else:
#             t_stride = (1,1,1)
#         s_stride = (1,stride,stride)
#         t_padding = (1,0,0)
#         s_padding = (0,padding,padding)
#         self.per_out_channels = per_out_channels
#         self.still  =  nn.Conv3d(in_channels,out_channels-4*per_out_channels,t_kernel_size,t_stride,t_padding,dilation,groups)
#         self.up     =  DirectionalConv(in_channels,per_out_channels,  t_kernel_size,t_stride,None,dilation,groups,bias,mode='up')
#         self.down   =  DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, None, dilation, groups,bias,mode='down')
#         self.right  =  DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, None, dilation, groups,bias,mode='right')
#         self.left   =  DirectionalConv(in_channels,per_out_channels,  t_kernel_size,t_stride,None,dilation,groups,bias,mode = 'left')
#         self.still_bn = nn.BatchNorm3d(out_channels-4*per_out_channels)
#         self.up_bn = nn.BatchNorm3d(per_out_channels)
#         self.down_bn = nn.BatchNorm3d(per_out_channels)
#         self.right_bn = nn.BatchNorm3d(per_out_channels)
#         self.left_bn = nn.BatchNorm3d(per_out_channels)
#         self.still_s = nn.Conv3d(out_channels-4*per_out_channels,out_channels-4*per_out_channels,s_kernel_size,s_stride,s_padding,bias=bias)
#         self.up_s = nn.Conv3d(per_out_channels, per_out_channels,s_kernel_size, s_stride, s_padding, bias=bias)
#         self.down_s = nn.Conv3d(per_out_channels, per_out_channels,s_kernel_size, s_stride, s_padding, bias=bias)
#         self.right_s = nn.Conv3d(per_out_channels, per_out_channels,s_kernel_size, s_stride, s_padding, bias=bias)
#         self.left_s = nn.Conv3d(per_out_channels, per_out_channels,s_kernel_size, s_stride, s_padding, bias=bias)
#         self.still_bn_s = nn.BatchNorm3d(out_channels-4*per_out_channels)
#         self.up_bn_s = nn.BatchNorm3d(per_out_channels)
#         self.down_bn_s = nn.BatchNorm3d(per_out_channels)
#         self.right_bn_s = nn.BatchNorm3d(per_out_channels)
#         self.left_bn_s = nn.BatchNorm3d(per_out_channels)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         x1 = self.relu(self.still_bn(self.still(x)))
#         x1 = self.relu(self.still_bn_s(self.still_s(x1)))
#         x2 = self.relu(self.up_bn(self.up(x)))
#         x2 = self.relu(self.up_bn_s(self.up_s(x2)))
#         x3 = self.relu(self.down_bn(self.down(x)))
#         x3 = self.relu(self.down_bn_s(self.down_s(x3)))
#         x4 = self.relu(self.right_bn(self.right(x)))
#         x4 = self.relu(self.right_bn_s(self.right_s(x4)))
#         x5 = self.relu(self.left_bn(self.left(x)))
#         x5 = self.relu(self.left_bn_s(self.left_s(x5)))
#         out = torch.cat([x1,x2,x3,x4,x5],1)
#         return out


# class MDConv(nn.Module):
#     def __init__(self, in_channels, out_channels,
#                  kernel_size=3, stride=1, padding=1 ,  dilation=1, groups=1, bias=False,ratial = 0.2,t_downsample = False):
#         super(MDConv, self).__init__()
#         if not torch.cuda.is_available():
#             raise EnvironmentError("only support for GPU mode")
#         per_out_channels = int((1-ratial)/4*out_channels)
#         t_kernel_size = (3, 1 , 1 )
#         s_kernel_size = (1,kernel_size,kernel_size)
#
#         if t_downsample:
#             t_stride = (stride,1,1)
#         else:
#             t_stride = (1,1,1)
#         s_stride = (1,stride,stride)
#         t_padding = (1,0,0)
#         s_padding = (0,padding,padding)
#         self.per_out_channels = per_out_channels
#         self.still  =  nn.Conv3d(in_channels,out_channels-4*per_out_channels,t_kernel_size,t_stride,t_padding,dilation,groups)
#         self.up     =  DirectionalConv(in_channels,per_out_channels,  t_kernel_size,t_stride,None,dilation,groups,bias,mode='up')
#         self.down   =  DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, None, dilation, groups,bias,mode='down')
#         self.right  =  DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, None, dilation, groups,bias,mode='right')
#         self.left   =  DirectionalConv(in_channels,per_out_channels,  t_kernel_size,t_stride,None,dilation,groups,bias,mode = 'left')
#         self.still_bn = nn.BatchNorm3d(out_channels-4*per_out_channels)
#         self.t_bn = nn.BatchNorm3d(out_channels)
#         self.spatial = nn.Conv3d(out_channels,out_channels,s_kernel_size,s_stride,s_padding,bias=bias)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         x1 = self.still(x)
#         x2 = self.up(x)
#         x3 = self.down(x)
#         x4 = self.right(x)
#         x5 = self.left(x)
#         t_out = torch.cat([x1,x2,x3,x4,x5],1)
#         t_out = self.relu(self.t_bn(t_out))
#         s_out = self.spatial(t_out)
#         return s_out

class MDConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, bias=False, ratial=0.2, t_downsample=False, first_conv=False):
        super(MDConv, self).__init__()
        if not torch.cuda.is_available():
            raise EnvironmentError("only support for GPU mode")
        per_out_channels = int((1 - ratial) / 4 * out_channels)
        t_kernel_size = (3, 1, 1)
        s_kernel_size = (1, kernel_size, kernel_size)

        if t_downsample:
            t_stride = (stride, 1, 1)
        else:
            t_stride = (1, 1, 1)
        s_stride = (1, stride, stride)
        t_padding = (1, 0, 0)
        s_padding = (0, padding, padding)
        self.per_out_channels = per_out_channels

        if first_conv:
            self.spatial = nn.Conv3d(in_channels, out_channels, s_kernel_size, s_stride, s_padding, bias=bias)
            self.bn = nn.BatchNorm3d(out_channels)
            in_channels = out_channels
        else:
            self.spatial = nn.Conv3d(in_channels, in_channels, s_kernel_size, s_stride, s_padding, bias=bias)
            self.bn = nn.BatchNorm3d(in_channels)
        self.still = nn.Conv3d(in_channels, out_channels - 4 * per_out_channels, t_kernel_size, t_stride, t_padding,
                               bias=bias)
        self.up = DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, None, bias=bias, mode='up')
        self.down = DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, None, bias=bias,
                                    mode='down')
        self.right = DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, None, bias=bias,
                                     mode='right')
        self.left = DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, None, bias=bias,
                                    mode='left')
        self.relu = nn.ReLU()

    def forward(self, x):
        s = self.relu(self.bn(self.spatial(x)))
        x1 = self.still(s)
        x2 = self.up(s)
        x3 = self.down(s)
        x4 = self.right(s)
        x5 = self.left(s)
        t = torch.cat([x1, x2, x3, x4, x5], 1)
        return t


# class MDConv(nn.Module):
#     def __init__(self, in_channels, out_channels,
#                  kernel_size=3, stride=1, padding=1 ,  bias=False,ratial = 0.2,t_downsample = False,first_conv=False):
#         super(MDConv, self).__init__()
#         if not torch.cuda.is_available():
#             raise EnvironmentError("only support for GPU mode")
#         per_out_channels = int((1-ratial)/4*out_channels)
#         t_kernel_size = (3, 1 , 1 )
#         s_kernel_size = (1,kernel_size,kernel_size)
#
#         if t_downsample:
#             t_stride = (stride,1,1)
#         else:
#             t_stride = (1,1,1)
#         s_stride = (1,stride,stride)
#         t_padding = (1,0,0)
#         s_padding = (0,padding,padding)
#         self.per_out_channels = per_out_channels
#
#         if first_conv:
#             self.spatial = nn.Conv3d(in_channels, out_channels, s_kernel_size, s_stride, s_padding, bias=bias)
#             self.bn = nn.BatchNorm3d(out_channels)
#             in_channels = out_channels
#         else:
#             self.spatial = nn.Conv3d(in_channels, in_channels, s_kernel_size, s_stride, s_padding, bias=bias)
#             self.bn = nn.BatchNorm3d(in_channels)
#         self.still  =  nn.Conv3d(in_channels,out_channels-4*per_out_channels,t_kernel_size,t_stride,t_padding,bias = bias)
#         self.up     =  nn.Conv3d(in_channels,per_out_channels,  t_kernel_size, t_stride,t_padding,bias=bias)
#         self.down   =  nn.Conv3d(in_channels, per_out_channels, t_kernel_size, t_stride, t_padding,bias=bias)
#         self.right  =  nn.Conv3d(in_channels, per_out_channels, t_kernel_size, t_stride, t_padding,bias=bias)
#         self.left   =  nn.Conv3d(in_channels,per_out_channels,  t_kernel_size,t_stride,t_padding,bias=bias)
#
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         s = self.relu(self.bn(self.spatial(x)))
#         x1 = self.still(s)
#         x2 = self.up(s)
#         x3 = self.down(s)
#         x4 = self.right(s)
#         x5 = self.left(s)
#         t = torch.cat([x1,x2,x3,x4,x5],1)
#         return t

class MDConv_first(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, bias=True, ratial=0.2):
        super(MDConv_first, self).__init__()
        if not torch.cuda.is_available():
            raise EnvironmentError("only support for GPU mode")
        self.per_in_channels = in_channels
        self.per_out_channels = out_channels
        self.still = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.up = DirectionalConv(in_channels, out_channels, kernel_size, stride, None, dilation, groups, bias,
                                  mode='up')
        self.down = DirectionalConv(in_channels, out_channels, kernel_size, stride, None, dilation, groups, bias,
                                    mode='down')
        self.right = DirectionalConv(in_channels, out_channels, kernel_size, stride, None, dilation, groups, bias,
                                     mode='right')
        self.left = DirectionalConv(in_channels, out_channels, kernel_size, stride, None, dilation, groups, bias,
                                    mode='left')

    def forward(self, x):
        x1 = self.still(x)
        x2 = self.up(x)
        x3 = self.down(x)
        x4 = self.right(x)
        x5 = self.left(x)
        out = torch.cat([x1, x2, x3, x4, x5], 1)
        return out

# z = torch.empty((1,1,3,3,3))
# z.fill_(1./27)
# print(z)

# input = torch.ones((1,1,3,3,3)).cuda()
# # weight = Parameter(torch.ones((1,1,3,3,3),requires_grad=True).cuda())
# # bias = Parameter(torch.ones((1),requires_grad=True).cuda())
#
# #test#####################
# conv = DirectionalConv(1,1,(3,1,1),(2,1,1),mode='right').cuda()
#
# init.constant(conv.weight,1)
# init.constant(conv.bias,0)
# out = conv.forward(input)
# print(out)

# input = torch.ones((1,1,3,3,3)).cuda()
# input2 = torch.ones((1,1,3,3,3)).cuda()
# x = torch.cat([input,input2],1)
# out = conv.forward(x)
# l = out.mean()
# l.backward()
# print(out)
# print(conv.weight)
# print(conv.weight.grad)
# out1 = MDConvFunction.apply(input,weight,bias,3,1,1,1,1)
# l = out1.mean()
# l.backward()
# print(weight.grad)
# from torch.autograd import grad
# input = torch.randn((8,3,32,56,56)).double().cuda()
# weight = Parameter(torch.randn(6,3,3,1,1,requires_grad=True).double().cuda())
# bias = Parameter(torch.ones((6),requires_grad=True).double().cuda())

# out = F.conv3d(test_input,weight,bias,(1,1,1),(1,1,1),(1,1,1),1)
# a1,a2,a3,b1,b2,b3 = 0,   1,   0,   0,   0,   0       down
# a1,a2,a3,b1,b2,b3 = 0,  -1, k[0]-1,0    0,   0        up
# a1,a2,a3,b1,b2,b3 = 0,   0,   0,   0,   1,   0       right
# a1,a2,a3,b1,b2,b3 = 0,   0,   0,   0,  -1,   k[0]-1  left
# k = (3,1,1)
# s = 1
# p = (1,0,1)
# d = 1
# g = 1
# a1,a2,a3,b1,b2,b3 = 0,0,0,0,-1,k[0]-1
# out = MDConvFunction.apply(input,weight,bias,k,s,p,d,g,a1,a2,a3,b1,b2,b3)
# out = F.conv3d(input,weight,bias,(1,1,1),(1,0,0),(1,1,1),1)

# print(out)
# l = out.mean()
# l.backward()
# x = torch.sum(out)
# l.backward()
# grad_output = grad(l,out)
# print(grad_output)
# l.backward(grad_output)
# print(weight.grad)
# print(weight.grad)


# print(torch.autograd.gradcheck(MDConvFunction.apply, (input,weight,bias,k,s,p,d,g,a1,a2,a3,b1,b2,b3), eps=1e-3))


# for i in tqdm(range(1000),total=1000):
#     out = MDConvFunction.apply(test_input, weight, bias, 3, 1, 1, 1, 1)
#     # out = F.conv3d(test_input, weight, bias, (1, 1, 1), (1, 1, 1), (1, 1, 1), 1)
#     l = out.mean()
#     l.backward()
