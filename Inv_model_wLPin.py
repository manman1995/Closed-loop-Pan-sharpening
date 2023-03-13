import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from Inv_modules import InvertibleConv1x1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(ResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))
        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out

class HinResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(HinResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(feature // 2, affine=True)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))

        out_1, out_2 = torch.chunk(residual, 2, dim=1)
        residual = torch.cat([self.norm(out_1), out_2], dim=1)

        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out

class INSG(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(INSG, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature*2, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature, channel_out, kernel_size=3, padding=1)
        #self.conv3 = nn.Conv2d(feature, channel_out, kernel_size=1)
        self.norm = nn.InstanceNorm2d(feature, affine=True)

        self.convf = nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1)

    def forward(self, x):

        residual = self.relu1(self.conv1(x))

        out_1 = self.norm(residual)
        out_2 = residual - self.norm(residual)

        #out_1, out_2 = torch.chunk(residual, 2, dim=1)
        residual = torch.cat([out_1, out_2], dim=1)

        residual = self.relu1(self.conv2(residual))

        out = self.conv3(residual)+self.convf(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        elif net_structure == 'Resnet':
            return ResBlock(channel_in, channel_out)
        elif net_structure == 'HinResnet':
            return HinResBlock(channel_in, channel_out)
        else:
            return None
    return constructor


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        #in_channels = 3
        self.invconv = InvertibleConv1x1(channel_num, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        if not rev:
            # invert1x1conv
            x, logdet = self.flow_permutation(x, logdet=0, rev=False)

            # split to 1 channel and 2 channel.
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

            y1 = x1 + self.F(x2)  # 1 channel
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
            out = torch.cat((y1, y2), 1)
        else:
            # split.
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

            x = torch.cat((y1, y2), 1)

            # inv permutation
            out, logdet = self.flow_permutation(x, logdet=0, rev=True)

        return out


class InvISPNet(nn.Module):
    def __init__(self, channel_in=4, channel_split_num=4, subnet_constructor=subnet('HinResnet'), block_num=8):
        super(InvISPNet, self).__init__()
        operations = []
        level = 3

        channel_num = channel_in + (level +1)  #total channels at input stage

        for j in range(block_num):
            b = InvBlock(subnet_constructor, channel_num, channel_split_num)  # one block is one flow step.
            operations.append(b)

        self.operations = nn.ModuleList(operations)
        self.initialize()

        self.pyin = Lap_Pyramid_Conv(num_high=level)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

    def forward(self, ms, pan, rev=False):
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        mHR = upsample(ms, M, N)

        x = mHR
        out = self.pyin.pyramid_decom(pan)
        for i in range(len(out)):
            x = torch.cat([x, out[i]], dim=1)

        if not rev:
            for op in self.operations:
                out = op.forward(x, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(x, rev)
        return out, out[:, :4, :, :]

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=device, channels=3): #cuda
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            diff = nn.functional.interpolate(diff, size=(img.shape[2], img.shape[3]))  #
            pyr.append(diff)
            current = down
        current = nn.functional.interpolate(current, size=(img.shape[2], img.shape[3]))
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image

if __name__ == '__main__':
    level =3
    pyin = Lap_Pyramid_Conv(num_high=level)
    net = InvISPNet(channel_in=3, level=level,block_num=8)
    print('#generator parameters:',sum(param.numel() for param in net.parameters()))
    x = torch.randn(2, 3, 128, 128)
    out = pyin.pyramid_decom(x)
    for i in range(len(out)):
        x =torch.cat([x,out[i]],dim=1)
    print(x.size())
    out = net(x)
    print(out.shape)