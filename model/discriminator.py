from model import *


class Discriminator(nn.Module):
    """
    Better use the Dual-Video-Discriminator
    """
    def __init__(self, out_dim=256, block_depth=3, latent_dim=256):
        super(Discriminator, self).__init__()
        self.dim = out_dim
        self.ld = latent_dim
        self.t = 16
        self.conv_block = nn.Sequential(ConvBlock3D(in_channel=3, depth=block_depth, channel=64),
                                        ConvBlock3D(in_channel=64, depth=block_depth, channel=128),
                                        ConvBlock3D(in_channel=128, depth=block_depth, channel=256),
                                        ConvBlock3D(in_channel=256, depth=block_depth, channel=256))
        self.linear = nn.Linear(out_dim * 8 * 8, latent_dim)
        self.linear1 = nn.Linear(latent_dim * self.t, latent_dim)
        # self.bn = nn.BatchNorm3d(latent_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        pass

    def forward(self, frames):
        if frames.size(2) == 3:
            frames = torch.transpose(frames, 1, 2)
        output = self.conv_block(frames)  # (batch)
        output = output.view(-1, self.t, self.dim * 8 * 8)
        output = self.relu(self.linear(output))
        output = output.view(-1, self.ld * self.t)
        output = self.linear1(output)
        output = self.sigmoid(output)
        return output


class CodeDiscriminator(nn.Module):
    """
    Better use the Dual-Video-Discriminator
    Add sigmoid in network.compute_adversarial_loss for WGAN convenient
    """

    def __init__(self, out_dim=512, block_depth=3, latent_dim=512):
        super(CodeDiscriminator, self).__init__()
        self.dim = out_dim
        self.ld = latent_dim
        self.t = 16
        self.conv_block = ConvBlock3D(in_channel=256, depth=block_depth, channel=512)

        self.linear = nn.Linear(out_dim * 4 * 4, latent_dim)
        self.linear1 = nn.Linear(latent_dim * self.t, latent_dim)
        # self.bn = nn.BatchNorm3d(latent_dim)
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        pass

    def forward(self, codes):  # codes-->(batch, 256, 16, 8, 8)
        if codes.size(2) == 3:
            codes = torch.transpose(codes, 1, 2)
        output = self.conv_block(codes)  # (batch, 512, 16, 4, 4)
        output = output.view(-1, self.t, self.dim * 4 * 4)
        output = self.relu(self.linear(output))
        output = output.view(-1, self.ld * self.t)
        output = self.linear1(output)
        # output = self.sigmoid(output)
        return output


if __name__ == '__main__':
    D = Discriminator()
    x = torch.randn(2, 3, 16, 128, 128)
    y = D(x)
    print(x.size())
    print(y.size())
