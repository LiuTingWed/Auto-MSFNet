import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
    'none': lambda in_C, out_C, stride, upsample, affine: Zero(stride, upsample=upsample),
    'skip_connect': lambda in_C, out_C, stride, upsample, affine: Identity(upsample=upsample),
    'sep_conv_3x3': lambda in_C, out_C, stride, upsample, affine: SepConv(in_C, out_C, 3, stride, 1, affine=affine,
                                                               upsample=upsample),
    'sep_conv_3x3_rp2': lambda in_C, out_C, stride, upsample, affine: SepConvDouble(in_C, out_C, 3, stride, 1,
                                                                                    affine=affine, upsample=upsample),
    'dil_conv_3x3': lambda in_C, out_C, stride, upsample, affine: DilConv(in_C, out_C, 3, stride, 2, 2, affine=affine,
                                                                          upsample=upsample),
    'dil_conv_3x3_rp2': lambda in_C, out_C, stride, upsample, affine: DilConvDouble(in_C, out_C, 3, stride, 2, 2,
                                                                                    affine=affine, upsample=upsample),
    'dil_conv_3x3_dil4': lambda in_C, out_C, stride, upsample, affine: DilConv(in_C, out_C, 3, stride, 4, 4,
                                                                               affine=affine, upsample=upsample),

    'conv_3x3': lambda in_C, out_C, stride, upsample, affine: Conv(in_C, out_C, 3, stride, 1, affine=affine,
                                                                   upsample=upsample),
    'conv_3x3_rp2': lambda in_C, out_C, stride, upsample, affine: ConvDouble(in_C, out_C, 3, stride, 1, affine=affine,
                                                                             upsample=upsample),

    'SpatialAttention': lambda in_C, out_C, stride, upsample, affine: SpatialAttention(in_C,7),
    'ChannelAttention': lambda in_C, out_C, stride, upsample, affine: ChannelAttention(in_C,16),

}


def conv3x3(in_planes, out_planes, stride):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio):
        super(ChannelAttention, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, self.in_channels // ratio)
        self.linear_2 = nn.Linear(self.in_channels // ratio, self.in_channels)
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
    def forward(self, input_):
        n_b, n_c, h, w = input_.size()

        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))

        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()
        out  = torch.mul(input_, feats)
        out = F.relu(self.bn1(self.conv1(out)), inplace=True)

        return out


class SpatialAttention(nn.Module):
    def __init__(self,in_C, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.in_channels = in_C
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv11 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False)
        self.bn11 = nn.BatchNorm2d(self.in_channels)
    def forward(self, x):
        input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        out  = input * x

        out = F.relu(self.bn11(self.conv11(out)), inplace=True)

        return out


class Conv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, upsample, affine=True):
        super(Conv, self).__init__()
        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        if self.upsample is True:
            x = self.up(x)
        return self.op(x)


class ConvDouble(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, upsample, affine=True):
        super(ConvDouble, self).__init__()

        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        if self.upsample is True:
            x = self.up(x)
        return self.op(x)


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, upsample, affine=True):
        super(DilConv, self).__init__()

        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        if self.upsample is True:
            x = self.up(x)
        return self.op(x)


class DilConvDouble(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, upsample, affine=True):
        super(DilConvDouble, self).__init__()
        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        if self.upsample is True:
            x = self.up(x)
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, upsample, affine=True):
        super(SepConv, self).__init__()

        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        if self.upsample is True:
            x = self.up(x)
        return self.op(x)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0

        self.up = nn.Sequential(
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.up(x)
        x = self.relu(x)
        out = self.conv_1(x)
        out = self.bn(out)
        return out


class SepConvDouble(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, upsample, affine=True):
        super(SepConvDouble, self).__init__()

        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        if self.upsample is True:
            x = self.up(x)
        return self.op(x)



class Identity(nn.Module):

    def __init__(self, upsample):
        super(Identity, self).__init__()
        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )

    def forward(self, x):
        if self.upsample == True:
            x = self.up(x)
        return x


class Zero(nn.Module):

    def __init__(self, stride, upsample):
        super(Zero, self).__init__()
        self.stride = stride
        self.upsample = upsample
        self.up = nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )

    def forward(self, x):
        if self.upsample == True:
            x = self.up(x)
        else:
            x = x.mul(0.)
        return x
        # return x[:,:,::self.stride,::self.stride].mul(0.)
