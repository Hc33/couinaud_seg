"""Encoder, decoder, and attention layers for the co-learning network."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_conv_stack(depth, num_channels):
    layers = []
    for _ in range(depth):
        layers.append(ConvBnAct3d(in_ch=num_channels, out_ch=num_channels))
    return nn.Sequential(*layers)


def passthrough(x):
    return x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ConvBnAct3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBnAct3d, self).__init__()
        self.conv = nn.Conv3d(in_channels= in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(num_features=out_ch)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InputTransition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InputTransition, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x


class DownTransition(nn.Module):
    """Downsampling followed by residual convolution."""
    def __init__(self, in_ch, out_ch, depth=3, dropout=False):
        super(DownTransition, self).__init__()
        self.down_conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.act1 = nn.ReLU(inplace=False)
        self.act2 = nn.ReLU(inplace=False)
        self.conv_block = make_conv_stack(depth, out_ch)
        self.short_dropout = nn.Dropout3d(p=0.25, inplace=False) if dropout else passthrough

    def forward(self, x):
        x = self.act1(self.bn1(self.down_conv(x)))
        x = self.short_dropout(x)
        x = torch.add(self.conv_block(x), x)
        return self.act2(x)


class UpsampleBlock(nn.Module):
    """
    upsample feature map with transpose convolution
    """
    def __init__(self, in_ch, out_ch, dropout=False):
        super(UpsampleBlock, self).__init__()
        self.up_conv =  nn.ConvTranspose3d(in_ch, out_ch//2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(out_ch//2)
        self.act1 = nn.ReLU(inplace=False)
        self.skip_dropout = nn.Dropout(p=0.25, inplace=False) if dropout else passthrough

    def forward(self, x, skip_x):
        x = self.act1(self.bn1(self.up_conv(x)))
        skip_x = self.skip_dropout(skip_x)
        x = torch.cat((x, skip_x), 1)
        return x


class DecoderFusionBlock(nn.Module):
    """Multiply the routed and decoder features, convolve, then add the decoder residual."""
    def __init__(self, in_ch, depth=3, dropout=False):
        super(DecoderFusionBlock, self).__init__()
        self.short_dropout = nn.Dropout(p=0.25, inplace=False) if dropout else passthrough
        self.conv_block = make_conv_stack(depth, in_ch)

    def forward(self, x, routed_features):
        short_skip = self.short_dropout(x)
        x = routed_features * x
        x = self.conv_block(x)
        x = short_skip + x
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fc1 = nn.Linear(in_features=out_ch, out_features=round(out_ch/16))
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(in_features=round(out_ch/16), out_features=out_ch)

    def forward(self, x):
        x_ca = F.adaptive_avg_pool3d(x, output_size=(1,1,1))
        x_ca = x_ca.view(x.size(0), -1)
        x_ca = self.relu(self.fc1(x_ca))
        x_ca = self.fc2(x_ca)
        x_ca = torch.sigmoid(x_ca)
        x_ca = x_ca.view(x_ca.size(0), x_ca.size(1), 1,1,1)
        return x * x_ca


class SpatialAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.compress = ChannelPool()
        kernel_size = 5
        self.spatial_1 = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_sa = self.compress(x)
        x_sa = self.spatial_1(x_sa)
        x_sa = torch.sigmoid(x_sa)
        return x * x_sa


class OutputTransition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(in_ch)
        self.act = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.act(self.bn(self.conv1(x)))
        x = self.conv2(x)
        return x
