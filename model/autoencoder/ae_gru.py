import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ConvGRU import ConvGRU
from model.Normalization import SpectralNorm
from model.Attention import SelfAttention

from model import ConvBlock2D, DeconvUpsamplingBlock2D


class GBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3, 3],
                 padding=1, stride=1, bn=True,
                 activation=F.relu6, upsample=False, downsample=True):
        super().__init__()

        # self.conv0 = SpectralNorm(nn.Conv2d(in_channel, out_channel,
        #                                     kernel_size, stride, padding,
        #                                     bias=True if bn else True))
        # self.conv1 = SpectralNorm(nn.Conv2d(out_channel, out_channel,
        #                                     kernel_size, stride, padding,
        #                                     bias=True if bn else True))
        self.conv0 = nn.Conv2d(in_channel, out_channel,
                                            kernel_size, stride, padding,
                                            bias=True if bn else True)
        self.conv1 = nn.Conv2d(out_channel, out_channel,
                                            kernel_size, stride, padding,
                                            bias=True if bn else True)
        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            # self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel,
            #                                       1, 1, 0))
            self.conv_sc = nn.Conv2d(in_channel, out_channel,
                                                  1, 1, 0)
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            # self.HyperBN = ConditionalNorm(in_channel, 148)
            # self.HyperBN_1 = ConditionalNorm(out_channel, 148)
            self.HyperBN = nn.BatchNorm2d(in_channel, eps=1e-6)
            self.HyperBN_1 = nn.BatchNorm2d(out_channel, eps=1e-6)

    def forward(self, input):

        out = input
        if self.bn:
            out = self.HyperBN(out)
        out = self.activation(out)
        if self.upsample:
            # TODO different form papers
            out = F.interpolate(out, scale_factor=2)
        out = self.conv0(out)

        if self.bn:
            out = self.HyperBN_1(out)
        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                # TODO different form papers
                skip = F.interpolate(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)

        else:
            skip = input

        return out + skip


class MotionEncoder(nn.Module):

    def __init__(self, chn=128, code_dim=1024):
        super().__init__()

        # self.pre_conv = nn.Sequential(SpectralNorm(nn.Conv2d(3, 2 * chn, 3, padding=1), ),
        #                               nn.ReLU(),
        #                               SpectralNorm(nn.Conv2d(2 * chn, 2 * chn, 3, padding=1), ),
        #                               nn.AvgPool2d(2))
        self.pre_conv = nn.Sequential(nn.Conv2d(3, 2 * chn, 3, padding=1),
                                      nn.BatchNorm2d(2*chn, affine=True),
                                      nn.ReLU(),
                                      nn.Conv2d(2 * chn, 2 * chn, 3, padding=1),
                                      nn.AvgPool2d(2))
        # self.pre_skip = SpectralNorm(nn.Conv2d(3, 2 * chn, 1))
        self.pre_skip = nn.Conv2d(3, 2 * chn, 1)

        self.conv1 = GBlock(2 * chn, 4 * chn, bn=False, upsample=False, downsample=True)
        self.attn = SelfAttention(4 * chn)
        self.conv2 = nn.Sequential(
            GBlock(4 * chn, 8 * chn, bn=False, upsample=False, downsample=True),
            GBlock(8 * chn, 16 * chn, bn=False, upsample=False, downsample=True),
            GBlock(16 * chn, 16 * chn, bn=False, upsample=False, downsample=True)
        )

        # self.linear = SpectralNorm(nn.Linear(16 * chn, 240))
        self.linear = nn.Linear(16 * chn, code_dim)

        # self.embed = nn.Embedding(n_class, 16 * chn)
        # self.embed.weight.data.uniform_(-0.1, 0.1)
        # self.embed = SpectralNorm(self.embed)

    def forward(self, x):
        # reshape input tensor from BxTxCxHxW to BTxCxHxW
        batch_size, T, C, W, H = x.size()
        x = x.view(batch_size * T, C, H, W)
        print('=======input: ', x[1, 1, :, :], x.size())

        out = self.pre_conv(x)
        print('=======pre_conv: ', out[1, 1, :, :], out.size())  # nan start !!!
        out = out + self.pre_skip(F.avg_pool2d(x, 2))
        print('=======pre_skip: ', out[1, 1, :, :], out.size())

        # reshape back to B x T x C x H x W

        # out = out.view(batch_size, T, -1, H // 2, W // 2)

        middle = self.conv1(out)  # BT x C x H x W
        print('================================middle:', middle[1, 1, :, :], middle.size())
        # out = out.permute(0, 2, 1, 3, 4) # B x C x T x H x W

        # out = self.attn(out)  # B x C x T x H x W  (should be utilized !!! but improve compatibility ?!)
        # out = out.permute(0, 2, 1, 3, 4).contiguous() # B x T x C x H x W

        out = self.conv2(middle)
        out = F.relu(out)
        # out = out.permute(0, 2, 1, 3, 4).contiguous()
        # out = out.view(out.size(0), out.size(1), -1)
        out = out.view(batch_size, T, out.size(1), out.size(2), -1)  # B x T x C x H x W

        # sum on H and W axis
        out = out.sum(-1)
        out = out.sum(-1)
        # sum on T axis
        out = out.sum(1)

        out_linear = self.linear(out).squeeze(1)
        print('===================================motion code: ', out_linear[1, :], out_linear.size())  # nan !
        # repeat class_id for each frame
        # TODO: test in case multi-class
        # class_id = class_id.view(-1, 1).repeat(1, T).view(-1)

        # embed = self.embed(class_id)

        # prod = (out * embed).sum(1)

        # out_linear = out_linear.view(-1, T)
        # prod = prod.view(-1, T)

        # score = (out_linear + prod).sum(1)
        return out_linear, middle  # + prod


class MotionEncoderGRU(nn.Module):
    def __init__(self, out_dim=1024, latent_dim=8, ch=32, sn=False):
        super(MotionEncoderGRU, self).__init__()

        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.ch = ch

        if sn:
            self.preconv = SpectralNorm(nn.Conv2d(3, 2 * ch, kernel_size=(3, 3), padding=1))
        else:
            self.preconv = nn.Sequential(nn.Conv2d(3, 2 * ch, kernel_size=(3, 3), padding=1),
                                         nn.BatchNorm2d(2*ch, affine=True))

        # self.conv = nn.ModuleList([
        #     GResBlock(2 * ch, 4 * ch, upsample_factor=1),
        #     GResBlock(4 * ch, 4 * ch, downsample_factor=2),
        #     ConvGRU(4 * ch, hidden_sizes=[4 * ch, 8 * ch, 4 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
        #     GResBlock(4 * ch, 8 * ch, upsample_factor=1),
        #     GResBlock(8 * ch, 8 * ch, downsample_factor=2),
        #     ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
        #     GResBlock(8 * ch, 8 * ch, upsample_factor=1),
        #     GResBlock(8 * ch, 8 * ch, downsample_factor=2),
        #     ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
        #     GResBlock(8 * ch, 8 * ch, upsample_factor=1),
        #     GResBlock(8 * ch, 8 * ch, downsample_factor=2),
        #     ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3)
        # ])
        self.conv = nn.ModuleList([
            ConvBlock2D(in_channel=2 * ch, depth=1, channel=4 * ch),
            ConvGRU(4 * ch, hidden_sizes=[4 * ch, 4 * ch], kernel_sizes=[3, 3], n_layers=2),
            ConvBlock2D(in_channel=4 * ch, depth=1, channel=8 * ch),
            ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
            ConvBlock2D(in_channel=8 * ch, depth=1, channel=8 * ch),
            ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
            ConvBlock2D(in_channel=8 * ch, depth=1, channel=8 * ch),
            ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2)
        ])

        self.linear = nn.Linear(latent_dim * latent_dim * 8 * ch, out_dim)
        # self.mu_linear = nn.Linear(out_dim, out_dim)
        # self.sigma_linear = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        batch_size, T, C, W, H = x.size()

        x = x.view(batch_size * T, C, H, W)

        y = self.preconv(x)  # (BT, 64, 128, 128)

        for k, conv in enumerate(self.conv):
            if isinstance(conv, ConvGRU):

                _, C, W, H = y.size()
                y = y.view(-1, T, C, W, H).contiguous()

                frame_list = []
                for i in range(T):
                    if k == 0:
                        if i == 0:
                            frame_list.append(conv(y))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y, frame_list[i - 1]))
                    else:
                        if i == 0:
                            frame_list.append(conv(y[:, 0, :, :, :].squeeze(1)))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y[:, i, :, :, :].squeeze(1), frame_list[i - 1]))

                frame_hidden_list = []
                for i in frame_list:
                    frame_hidden_list.append(i[-1].unsqueeze(0))
                y = torch.cat(frame_hidden_list, dim=0)  # T x B x ch x ld x ld
                y = y.permute(1, 0, 2, 3, 4).contiguous()  # B x T x ch x ld x ld
                # print(y.size())
                B, T, C, W, H = y.size()

                if k == len(self.conv) - 1:
                    y = y[:, T - 1, :, :, :]
                y = y.view(-1, C, W, H)  # B * 256 * 16 * 16

            elif not isinstance(conv, ConvGRU):
                y = conv(y)

        y = y.view(-1, C * W * H)
        y = self.linear(y)

        return y


class GResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=None,
                 padding=1, stride=1, bn=True,
                 activation=F.relu, upsample_factor=2, downsample_factor=1):
        super().__init__()

        self.upsample_factor = upsample_factor if downsample_factor is 1 else 1
        self.downsample_factor = downsample_factor
        self.activation = activation
        self.bn = bn if downsample_factor is 1 else False

        if kernel_size is None:
            kernel_size = [3, 3]

        # self.conv0 = SpectralNorm(nn.Conv2d(in_channel, out_channel,
        #                                     kernel_size, stride, padding,
        #                                     bias=True if bn else True))
        # self.conv1 = SpectralNorm(nn.Conv2d(out_channel, out_channel,
        #                                     kernel_size, stride, padding,
        #                                     bias=True if bn else True))
        self.conv0 = nn.Conv2d(in_channel, out_channel,
                                            kernel_size, stride, padding,
                                            bias=True if bn else True)
        self.conv1 = nn.Conv2d(out_channel, out_channel,
                                            kernel_size, stride, padding,
                                            bias=True if bn else True)

        self.skip_proj = True
        # self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
        self.conv_sc = nn.Conv2d(in_channel, out_channel, 1, 1, 0)

        # if in_channel != out_channel or upsample_factor or downsample_factor:
        #     self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
        #     self.skip_proj = True

        if bn:
            #     self.CBNorm1 = ConditionalNorm(in_channel, n_class)  # TODO 2 x noise.size[1]
            #     self.CBNorm2 = ConditionalNorm(out_channel, n_class)
            self.CBNorm1 = nn.BatchNorm2d(in_channel, affine=True)
            self.CBNorm2 = nn.BatchNorm2d(out_channel, affine=True)
            self.CBNorm3 = nn.BatchNorm2d(out_channel, affine=True)

    def forward(self, x):

        # The time dimension is combined with the batch dimension here, so each frame proceeds
        # through the blocks independently
        BT, C, W, H = x.size()
        out = x

        if self.bn:
            out = self.CBNorm1(out)

        out = self.activation(out)

        if self.upsample_factor != 1:
            out = F.interpolate(out, scale_factor=self.upsample_factor)

        out = self.conv0(out)

        if self.bn:
            out = out.view(BT, -1, W * self.upsample_factor, H * self.upsample_factor)
            out = self.CBNorm2(out)

        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample_factor != 1:
            out = F.avg_pool2d(out, self.downsample_factor)

        if self.skip_proj:
            skip = x
            if self.upsample_factor != 1:
                skip = F.interpolate(skip, scale_factor=self.upsample_factor)
            skip = self.conv_sc(skip)
            if self.downsample_factor != 1:
                skip = F.avg_pool2d(skip, self.downsample_factor)
        else:
            skip = x

        y = out + skip
        y = y.view(
            BT, -1,
            W * self.upsample_factor // self.downsample_factor,
            H * self.upsample_factor // self.downsample_factor
        )

        return y


class DecoderGRU(nn.Module):
    def __init__(self, in_dim=1024, latent_dim=8, ch=32, n_frames=16):
        super(DecoderGRU, self).__init__()

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.ch = ch
        self.n_frames = n_frames

        self.double_transform = nn.Linear(in_dim, in_dim * 2)
        self.affine_transfrom = nn.Linear(in_dim, latent_dim * latent_dim * 8 * ch)

        # self.conv = nn.ModuleList([
        #     ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
        #     # ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
        #     GResBlock(8 * ch, 8 * ch, upsample_factor=1),
        #     GResBlock(8 * ch, 8 * ch),
        #     ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
        #     # ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
        #     GResBlock(8 * ch, 8 * ch, upsample_factor=1),
        #     GResBlock(8 * ch, 8 * ch),
        #     ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
        #     # ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
        #     GResBlock(8 * ch, 8 * ch, upsample_factor=1),
        #     GResBlock(8 * ch, 4 * ch),
        #     ConvGRU(4 * ch, hidden_sizes=[4 * ch, 8 * ch, 4 * ch], kernel_sizes=[3, 5, 5], n_layers=3),
        #     # ConvGRU(4 * ch, hidden_sizes=[4 * ch, 4 * ch], kernel_sizes=[3, 5], n_layers=2),
        #     GResBlock(4 * ch, 4 * ch, upsample_factor=1),
        #     GResBlock(4 * ch, 2 * ch)
        # ])
        self.conv = nn.ModuleList([
            ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
            DeconvUpsamplingBlock2D(in_channel=8 * ch, depth=1, channels=8 * ch),
            ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
            DeconvUpsamplingBlock2D(in_channel=8 * ch, depth=1, channels=8 * ch),
            ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
            DeconvUpsamplingBlock2D(in_channel=8 * ch, depth=1, channels=4 * ch),
            ConvGRU(4 * ch, hidden_sizes=[4 * ch, 4 * ch], kernel_sizes=[3, 5], n_layers=2),
            DeconvUpsamplingBlock2D(in_channel=4 * ch, depth=1, channels=2 * ch),
        ])

        # self.colorize = SpectralNorm(nn.Conv2d(2 * ch, 3, kernel_size=(3, 3), padding=1))
        self.colorize = nn.Conv2d(2 * ch, 3, kernel_size=(3, 3), padding=1)
        self.BN = nn.BatchNorm2d(3, affine=True)

    def forward(self, x):
        code = x
        y = self.affine_transfrom(code)  # B x (1 x ld x ch)

        y = y.view(-1, 8 * self.ch, self.latent_dim, self.latent_dim)  # B x ch x ld x ld

        for k, conv in enumerate(self.conv):
            if isinstance(conv, ConvGRU):

                if k > 0:
                    _, C, W, H = y.size()
                    y = y.view(-1, self.n_frames, C, W, H).contiguous()

                frame_list = []
                for i in range(self.n_frames):
                    if k == 0:
                        if i == 0:
                            frame_list.append(conv(y))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y, frame_list[i - 1]))
                    else:
                        if i == 0:
                            frame_list.append(conv(y[:, 0, :, :, :].squeeze(1)))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y[:, i, :, :, :].squeeze(1), frame_list[i - 1]))
                frame_hidden_list = []
                for i in frame_list:
                    frame_hidden_list.append(i[-1].unsqueeze(0))
                y = torch.cat(frame_hidden_list, dim=0)  # T x B x ch x ld x ld

                y = y.permute(1, 0, 2, 3, 4).contiguous()  # B x T x ch x ld x ld
                # print(y.size())
                B, T, C, W, H = y.size()
                y = y.view(-1, C, W, H)

            elif not isinstance(conv, GResBlock):
                # condition = torch.cat([noise_emb, class_emb], dim=1)
                # condition = condition.repeat(self.n_frames, 1)
                y = conv(y)  # BT, C, W, H
            # print(k, y.size())

        y = F.relu(y)
        y = self.BN(self.colorize(y))
        y = torch.tanh(y)

        BT, C, W, H = y.size()
        y = y.view(-1, self.n_frames, C, W, H)  # B, T, C, W, H

        return y


if __name__ == '__main__':
    # x = torch.randn([2, 4, 3, 128, 128])
    # generator = MotionEncoderGRU(n_frames=4)
    # y = generator(x)
    # print(y.size())

    y = torch.randn([1, 1024])  # (batch_size * t, dim)
    decoder = DecoderGRU()
    z = decoder(y)
    print(z.size())
