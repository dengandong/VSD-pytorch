from model import *


class MotionEncoderC2D(nn.Module):
    def __init__(self, out_dim=256, block_depth=3, latent_dim=256):
        super(MotionEncoderC2D, self).__init__()
        self.conv_block = nn.Sequential(ConvBlock2D(in_channel=3, depth=block_depth, channel=64),
                                        ConvBlock2D(in_channel=64, depth=block_depth, channel=128),
                                        ConvBlock2D(in_channel=128, depth=block_depth, channel=256),
                                        ConvBlock2D(in_channel=256, depth=block_depth, channel=256))

        self.linear = nn.Linear(out_dim * 8 * 8, latent_dim)
        # self.linear_mu = nn.Linear(latent_dim, latent_dim)
        # self.linear_logsigma = nn.Linear(latent_dim, latent_dim)

        # self.conv_mu = ConvBNRelu3D(in_channel=256, out_channel=256)
        # self.conv_logsigma = ConvBNRelu3D(in_channel=256, out_channel=256)
        pass

    def forward(self, input_frames, mode='conv'):
        # reshape btchw --> (b*t)chw
        if input_frames.size(1) == 3:
            input_frames = torch.transpose(input_frames, 1, 2)
        batch_size, T, C, W, H = input_frames.size()
        input_frames = input_frames.contiguous().view(batch_size * T, C, H, W)

        code = self.conv_block(input_frames)  # (batch*16, 256, 8, 8)

        if mode == 'conv':
            # mu = self.conv_mu(code)
            # log_sigma = self.conv_logsigma(code)
            pass
        else:
            _, c, t, w, h = code.size()
            code = code.view(-1, t, c * w * h)
            code = self.linear(code)
            # mu = self.linear_mu(code)
            # log_sigma = self.linear_logsigma(code)
        _, c, w, h = code.size()
        code = code.view(batch_size, T, c, w, h)
        code = torch.transpose(code, 1, 2)
        return code  # , mu, log_sigma


class ContentEncoder(nn.Module):
    def __init__(self, block_depth=3, latent_dim=256):
        super(ContentEncoder, self).__init__()
        self.encoder = nn.Sequential(ConvBlock2D(in_channel=3, depth=block_depth, channel=64),
                                     ConvBlock2D(in_channel=64, depth=block_depth, channel=128),
                                     ConvBlock2D(in_channel=128, depth=block_depth, channel=256),
                                     ConvBlock2D(in_channel=256, depth=block_depth, channel=256))  # 256, 8, 8
        self.linear = nn.Linear(256 * 8 * 8, latent_dim)

        self.linear1 = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(DeconvUpsamplingBlock2D(in_channel=256, depth=block_depth, channels=256),
                                     DeconvUpsamplingBlock2D(in_channel=256, depth=block_depth, channels=128),
                                     DeconvUpsamplingBlock2D(in_channel=128, depth=block_depth, channels=64),
                                     DeconvUpsamplingBlock2D(in_channel=64, depth=block_depth, channels=32))

        self.colorize = ConvBlock2D(in_channel=32, depth=1, channel=3, skip=False)

        pass

    def forward(self, frames, mode='conv'):
        """
        eliminate the temporal feature of the input frames by global pooling on time axis
        """
        # global low_level_representation, semantic_representation
        reduce_frames = frames[:, 0]
        assert reduce_frames.size(1) == 3, print('got wrong shape: ', reduce_frames.size())
        content_code = self.encoder(reduce_frames)
        if mode == 'conv':
            x = content_code
            pass
        else:
            _, c, h, w = content_code.size()
            content_code = content_code.view(-1, c * h * w)
            content_code = self.linear(content_code)
            # back to original shape
            x = self.linear1(content_code)
            x = F.relu(x)
            x = x.view(-1, c, h, w)
        x = self.decoder(x)
        reduce_reconstruction = self.colorize(x)

        content_code = time_axis_expand(content_code, 16)
        return content_code, reduce_frames, reduce_reconstruction


class VariationalContentEncoder(nn.Module):
    def __init__(self, block_depth=3, latent_dim=256):
        super(VariationalContentEncoder, self).__init__()
        self.encoder = nn.Sequential(ConvBlock2D(in_channel=3, depth=block_depth, channel=64),
                                     ConvBlock2D(in_channel=64, depth=block_depth, channel=128),
                                     ConvBlock2D(in_channel=128, depth=block_depth, channel=256),
                                     ConvBlock2D(in_channel=256, depth=block_depth, channel=256))  # 256, 8, 8
        # self.linear = nn.Linear(256 * 8 * 8, latent_dim)
        #
        # self.linear1 = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(DeconvUpsamplingBlock2D(in_channel=256, depth=block_depth, channels=256),
                                     DeconvUpsamplingBlock2D(in_channel=256, depth=block_depth, channels=128),
                                     DeconvUpsamplingBlock2D(in_channel=128, depth=block_depth, channels=64),
                                     DeconvUpsamplingBlock2D(in_channel=64, depth=block_depth, channels=32))

        self.conv_mu = ConvBNRelu2D(in_channel=256, out_channel=256)
        self.conv_sigma = ConvBNRelu2D(in_channel=256, out_channel=256)

        self.colorize = ConvBlock2D(in_channel=32, depth=1, channel=3, skip=False)

        pass

    def forward(self, img, eps, mode='conv'):
        """
        eliminate the temporal feature of the input frames by global pooling on time axis
        """
        content_code = self.encoder(img)
        if mode == 'conv':
            x = content_code
            x_mu = self.conv_mu(x)
            x_logsigma = self.conv_sigma(x)
            pass
        else:
            _, c, h, w = content_code.size()
            x = content_code.view(-1, c * h * w)
            # content_code = self.linear(content_code)
            # back to original shape
            # x = self.linear1(content_code)
            x = F.relu(x)
            x = x.view(-1, c, h, w)
        x = x_mu + torch.exp(x_logsigma / 2) * eps
        x = self.decoder(x)
        img_reconstruction = self.colorize(x)

        x_mu = time_axis_expand(x_mu, 16)
        x_logsigma = time_axis_expand(x_logsigma, 16)
        return x_mu, x_logsigma, img_reconstruction


class DecoderC2D(nn.Module):
    def __init__(self, block_depth=3, mode='interpolate'):
        super(DecoderC2D, self).__init__()
        self.linear = nn.Linear(256, 256 * 8 * 8)
        if mode == 'interpolate':
            self.deconv_block = nn.Sequential(DeconvUpsamplingBlock2D(in_channel=256, depth=block_depth, channels=256),
                                              DeconvUpsamplingBlock2D(in_channel=256, depth=block_depth, channels=128),
                                              DeconvUpsamplingBlock2D(in_channel=128, depth=block_depth, channels=64),
                                              DeconvUpsamplingBlock2D(in_channel=64, depth=block_depth, channels=32))
        elif mode == 'transpose':
            self.deconv_block = nn.Sequential(DeConvBlock2D(in_channel=256, depth=block_depth, channels=256),
                                              DeConvBlock2D(in_channel=256, depth=block_depth, channels=128),
                                              DeConvBlock2D(in_channel=128, depth=block_depth, channels=64),
                                              DeConvBlock2D(in_channel=64, depth=block_depth, channels=32))
        self.colorize = ConvBlock2D(in_channel=32, depth=1, channel=3, skip=False)
        pass

    def forward(self, code, mode='conv'):
        batch_size, c, T, h, w = code.size()
        code = code.view(batch_size * T, c, h, w)
        if mode == 'conv':
            x = code
            pass
        else:
            x = self.linear(code)
            x = x.view(-1, 256, 16, 8, 8)
        x = self.deconv_block(x)
        reconstruction = self.colorize(x)

        _, C, H, W = reconstruction.size()
        reconstruction = reconstruction.view(batch_size, T, C, H, W)
        return reconstruction


if __name__ == '__main__':
    # x = torch.randn(2, 16, 3, 128, 128)
    # encoder = MotionEncoderC2D()
    # y = encoder(x)
    # print(y.size())

    y = torch.randn(2, 16, 256, 8, 8)
    decoder = DecoderC2D()
    z = decoder(y)
    print(z.size())
