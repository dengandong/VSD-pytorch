from model import *


class SimpleSD(nn.Module):
    def __init__(self, ch=32, ld=4, n_classes=101):
        super(SimpleSD, self).__init__()

        self.latent_dim = ld
        self.channel = ch

        self.conv = nn.Sequential(ConvBlock2D(in_channel=3, depth=3, channel=ch),
                                  ConvBlock2D(in_channel=ch, depth=3, channel=2 * ch),
                                  ConvBlock2D(in_channel=2 * ch, depth=3, channel=4 * ch),
                                  ConvBlock2D(in_channel=4 * ch, depth=3, channel=8 * ch),
                                  ConvBlock2D(in_channel=8 * ch, depth=3, channel=8 * ch))
        self.linear = nn.Linear(8 * ch, 1)
        self.relu = nn.LeakyReLU()
        self.emb = nn.Embedding(n_classes, 8 * ch)
        # self.emb.weight.data.uniform_(-0.1, 0.1)

        pass

    def forward(self, x, class_id):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)

        x = self.conv(x)
        x = torch.sum(x, dim=(2, 3))  # BT x (8 * ch)

        out_x = self.linear(x)  # BT x 1

        class_id = class_id.view(-1, 1).repeat(1, T).view(-1)  # B --> BT
        class_emb = self.emb(class_id)  # BT x (8 * ch)

        prod = x * class_emb  # BT x (8 * ch)
        prod = torch.sum(prod, 1)  # BT x 1
        return out_x + prod


class SimpleTD(nn.Module):
    def __init__(self, ch=32, ld=4, n_classes=101):
        super(SimpleTD, self).__init__()

        self.latent_dim = ld
        self.channel = ch

        self.conv = nn.Sequential(ConvBlock3D(in_channel=3, depth=3, channel=ch),
                                  ConvBlock3D(in_channel=ch, depth=3, channel=2 * ch),
                                  ConvBlock3D(in_channel=2 * ch, depth=3, channel=4 * ch),
                                  ConvBlock3D(in_channel=4 * ch, depth=3, channel=8 * ch),
                                  ConvBlock3D(in_channel=8 * ch, depth=3, channel=8 * ch))
        self.linear = nn.Linear(8 * ch, 1)
        self.relu = nn.LeakyReLU()
        self.emb = nn.Embedding(n_classes, 8 * ch)
        # self.emb.weight.data.uniform_(-0.1, 0.1)

        pass

    def forward(self, x, class_id):
        if x.size(2) == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.conv(x)
        B, c, T, h, w = x.size()
        x = x.view(B * T, c, h, w)
        x = torch.sum(x, dim=(2, 3))  # BT x (8 * ch)

        out_x = self.linear(x)  # BT x 1

        class_id = class_id.view(-1, 1).repeat(1, T).view(-1)  # B --> BT
        class_emb = self.emb(class_id)  # BT x (8 * ch)

        prod = x * class_emb  # BT x (8 * ch)
        prod = torch.sum(prod, 1)  # BT x 1
        return out_x + prod


if __name__ == '__main__':
    batch_size = 4
    x = torch.randn(batch_size, 16, 3, 128, 128)
    class_id = torch.randint(0, 101, (batch_size, ))

    D = SimpleSD()
    out = D(x, class_id)
    out = out.mean()
    print(out)
