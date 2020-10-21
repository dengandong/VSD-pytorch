import torch
import torch.nn.functional as F
import torch.nn as nn


def time_axis_expand(x, k=16):
    x_unsqueeze = torch.unsqueeze(x, 2)
    if len(x.size()) == 2:
        x_expand = x_unsqueeze.expand(-1, k, -1)
    elif len(x.size()) == 4:
        x_expand = x_unsqueeze.expand(-1, -1, k, -1, -1)
        pass
    return x_expand


class ConvBNRelu2D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, is_pooling=False):
        super(ConvBNRelu2D, self).__init__()
        self.is_pooling = is_pooling
        if kernel_size == 1:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=0)
        elif kernel_size == 3:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channel, affine=True)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        pass

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.is_pooling:
            x = self.pool(x)
        return x


class DeconvBNRelu2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DeconvBNRelu2D, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, padding=1, stride=2)
        self.conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel, affine=True)
        self.relu = nn.LeakyReLU()
        pass

    def forward(self, x):
        x = self.deconv(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBlock2D(nn.Module):
    # TODO
    """
    the structure should be more sophisticated
    input size = [batch, 3, time_step, 128, 128]
    """

    def __init__(self, in_channel=3, depth=3, channel=32, skip=True):
        super(ConvBlock2D, self).__init__()

        self.depth = depth
        self.skip = skip

        if type(channel) != list:
            self.channel = depth * [channel]
        else:
            assert len(channel) == depth, 'the length of channel must match depth!'
            self.channel = channel

        self.conv_list = nn.ModuleList()
        if depth == 1:
            self.conv_list.append(ConvBNRelu2D(in_channel, self.channel[0], is_pooling=True))
        else:
            for i in range(depth):
                if i == 0:
                    self.conv_list.append(ConvBNRelu2D(in_channel, self.channel[0], is_pooling=False))
                elif i == depth-1:
                    self.conv_list.append(ConvBNRelu2D(self.channel[i-1], self.channel[i], is_pooling=True))
                else:
                    self.conv_list.append(ConvBNRelu2D(self.channel[i-1], self.channel[i], is_pooling=False))
        if skip:
            self.skip_conv = ConvBNRelu2D(in_channel, self.channel[-1], kernel_size=1, is_pooling=True)
        pass

    def forward(self, x):
        if self.skip:
            skip = x
            skip = self.skip_conv(skip)

        for i in range(self.depth):
            conv = self.conv_list[i]
            x = conv(x)
        if self.skip:
            x = skip + x
        return x


class DeConvBlock2D(nn.Module):
    # TODO
    """
    the structure should be more sophisticated
    input size = [batch, 128, time_step, 32, 32]
    """

    def __init__(self, in_channel, depth=3, channels=32):
        super(DeConvBlock2D, self).__init__()
        self.depth = depth

        if type(channels) != list:
            self.channels = [channels] * depth
        else:
            assert len(channels) == depth, 'the length of channels must match depth'
            self.channels = channels

        self.conv_list = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.conv_list.append(DeconvBNRelu2D(in_channel, self.channels[0]))
            else:
                self.conv_list.append(DeconvBNRelu2D(self.channels[i-1], self.channels[i]))
        pass

    def forward(self, x):
        for i in range(self.depth):
            deconv = self.conv_list[i]
            x = deconv(x)
        return x


class DeconvUpsamplingBlock2D(nn.Module):
    def __init__(self, in_channel, depth=3, channels=32):
        super(DeconvUpsamplingBlock2D, self).__init__()

        self.depth = depth

        if type(channels) != list:
            self.channels = [channels] * depth
        else:
            assert len(channels) == depth, 'the length of channels must match depth'
            self.channels = channels

        self.conv_list = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.conv_list.append(ConvBNRelu2D(in_channel, self.channels[0]))
            else:
                self.conv_list.append(ConvBNRelu2D(self.channels[i-1], self.channels[i]))

        self.skip_conv = ConvBNRelu2D(in_channel, self.channels[-1], kernel_size=1, is_pooling=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)

        skip = x
        skip = self.skip_conv(skip)

        for i in range(self.depth):
            conv = self.conv_list[i]
            x = conv(x)
        x = skip + x
        return x


class ConvBNRelu3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, is_pooling=False):
        super(ConvBNRelu3D, self).__init__()
        self.is_pooling = is_pooling
        if kernel_size == 1:
            self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=0, stride=1)
        elif kernel_size == 3:
            self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=1, stride=1)
        self.bn = nn.BatchNorm3d(out_channel, affine=True)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        pass

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.is_pooling:
            x = self.pool(x)
        return x


class DeconvBNRelu3D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DeconvBNRelu3D, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channel, out_channel,
                                         kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(out_channel, affine=True)
        self.relu = nn.LeakyReLU()
        pass

    def forward(self, x):
        x = self.deconv(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBlock3D(nn.Module):
    # TODO
    """
    the structure should be more sophisticated
    input size = [batch, 3, time_step, 128, 128]
    """
    def __init__(self, in_channel=3, depth=3, channel=64, skip=True):
        super(ConvBlock3D, self).__init__()

        self.depth = depth
        self.skip = skip
        if type(channel) != list:
            self.channel = depth * [channel]
        else:
            assert len(channel) == depth, 'the length of channel must match depth!'
            self.channel = channel

        self.conv_list = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.conv_list.append(ConvBNRelu3D(in_channel, self.channel[0], is_pooling=False))
            elif i == depth-1:
                self.conv_list.append(ConvBNRelu3D(self.channel[i-1], self.channel[i], is_pooling=True))
            else:
                self.conv_list.append(ConvBNRelu3D(self.channel[i-1], self.channel[i], is_pooling=False))
        if skip:
            self.skip_conv = ConvBNRelu3D(in_channel, self.channel[-1], kernel_size=1, is_pooling=True)
        pass

    def forward(self, x):
        if self.skip:
            skip = x
            skip = self.skip_conv(skip)
        for i in range(self.depth):
            conv = self.conv_list[i]
            x = conv(x)
        if self.skip:
            x = skip + x
        return x


class DeConvBlock3D(nn.Module):
    # TODO
    """
    the structure should be more sophisticated
    input size = [batch, 128, time_step, 32, 32]
    """
    def __init__(self, in_channel, depth=3, channels=32, skip=True):
        super(DeConvBlock3D, self).__init__()

        self.depth = depth
        self.skip = skip

        if type(channels) != list:
            self.channels = [channels] * depth
        else:
            assert len(channels) == depth, 'the length of channels must match depth'
            self.channels = channels

        self.conv_list = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.conv_list.append(DeconvBNRelu3D(in_channel, channels[0]))
            else:
                self.conv_list.append(DeconvBNRelu3D(channels[i-1], channels[i]))

        self.skip_conv = ConvBNRelu3D(in_channel, self.channels[-1], kernel_size=1, is_pooling=False)

        pass

    def forward(self, x):
        out = x
        for i in range(self.depth):
            conv = self.conv_list[i]
            out = conv(out)
        if self.skip:
            skip = self.skip_conv(x)
            out = skip + out
        return out


class DeconvUpsamplingBlock3D(nn.Module):
    def __init__(self, in_channel, depth=3, channels=32):
        super(DeconvUpsamplingBlock3D, self).__init__()

        self.depth = depth

        if type(channels) != list:
            self.channels = [channels] * depth
        else:
            assert len(channels) == depth, 'the length of channels must match depth'
            self.channels = channels

        self.conv_list = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.conv_list.append(ConvBNRelu3D(in_channel, self.channels[0]))
            else:
                self.conv_list.append(ConvBNRelu3D(self.channels[i-1], self.channels[i]))

        self.skip_conv = ConvBNRelu3D(in_channel, self.channels[-1], kernel_size=1, is_pooling=False)

    def forward(self, x):
        t, w, h = x.size()[2:]
        x = F.interpolate(x, size=(t, w*2, h*2))
        skip = x
        skip = self.skip_conv(skip)

        for i in range(self.depth):
            conv = self.conv_list[i]
            x = conv(x)

        x = skip + x
        return x


if __name__ == '__main__':
    x = torch.randn(16, 256, 8, 8)
    print(x.size())
    # conv3d = nn.Conv3d(16, 32, 1)
    conv3d = DeconvUpsamplingBlock2D(256, 1, 256)
    y = conv3d(x)
    print(y.size())
