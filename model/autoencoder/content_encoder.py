import torch
import torch.nn as nn
import torch.nn.functional as F
from model.autoencoder.ae_gru import GBlock
from model.Normalization import SpectralNorm


class ContentEncoder2(nn.Module):
    def __init__(self, chn=32, code_dim=1024):
        super(ContentEncoder2, self).__init__()
        self.conv1 = GBlock(3, 2 * chn, bn=True, upsample=False, downsample=True)
        self.conv2 = nn.Sequential(
            GBlock(2 * chn, 4 * chn, bn=True, upsample=False, downsample=True),
            GBlock(4 * chn, 8 * chn, bn=True, upsample=False, downsample=True),
            GBlock(8 * chn, 8 * chn, bn=True, upsample=False, downsample=True)
        )
        # self.linear = SpectralNorm(nn.Linear(8 * chn, 120))
        self.linear = nn.Linear(8 * chn, code_dim)

        # self.linear1 = SpectralNorm(nn.Linear(120, 8 * 8 * 8 * chn))
        self.linear1 = nn.Linear(code_dim, 8 * 8 * 8 * chn)
        self.conv3 = nn.Sequential(
            GBlock(8 * chn, 8 * chn, bn=True, upsample=True, downsample=False),
            GBlock(8 * chn, 4 * chn, bn=True, upsample=True, downsample=False),
            GBlock(4 * chn, 2 * chn, bn=True, upsample=True, downsample=False)
        )  # print the weight of this block !!!
        self.conv4 = GBlock(2 * chn, 3, bn=True, upsample=True, downsample=False)

    def forward(self, x):
        batch_size, T, C, W, H = x.size()
        # batch_size, C, W, H = x.size()

        gap = torch.mean(x, 1)  # B x C x H x W
        # gap = x
        # print('===gap: ', gap[1, 1, :, :], gap.size())

        # encode part
        out = self.conv1(gap)  # nan err
        # print('====conv1: ', out[1, 1, :, :], out.size())
        out = self.conv2(out)
        # print('====conv2: ', out[1, 1, :, :], out.size())
        out = out.view(batch_size, out.size(1), out.size(2), -1)  # B x C x H x W
        # sum on H and W axis
        out = out.sum(-1)
        # print('====sum1: ', out[1, 1, :], out.size())
        out = out.sum(-1)  # after this use bn ?
        # print('====sum2: ', out[1, :], out.size())
        code = self.linear(out)  # 120
        # print('====content code: ', code[1, :], code.size())

        # decode part
        out1 = self.linear1(code)  # 16384
        # print('linear code:', out1[1, :], out1.size())
        out1 = F.relu6(out1)
        out1 = out1.view(batch_size, -1, 8, 8)
        out1 = self.conv3(out1)
        # print('conv3 code:', out1[1, 1, :, :], out1.size())
        out1 = self.conv4(out1)
        # print('conv4 code:', out1[1, 1, :, :], out1.size())
        return code, gap, out1

