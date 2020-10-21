from model import *
from torch.autograd import Variable


class SimpleG(nn.Module):
    def __init__(self, n_class=101, n_dim=120, n_frames=16, ld=4, ch=32, cell_mode='lstm'):
        super(SimpleG, self).__init__()

        self.n_frames = n_frames
        self.latent_dim = ld
        self.channel = ch
        self.cell_mode = cell_mode

        self.class_emb = nn.Embedding(n_class, n_dim)
        if cell_mode == 'lstm':
            self.initial_generator = nn.LSTMCell(2 * n_dim, 2 * n_dim)
        elif cell_mode == 'gru':
            self.initial_generator = nn.GRUCell(2 * n_dim, 2 * n_dim)
        else:
            raise NotImplementedError('the {} should be defined'.format(cell_mode))

        self.transform = nn.Linear(n_dim * 2, 8 * ch * ld * ld)
        self.conv = nn.ModuleList([DeconvUpsamplingBlock2D(in_channel=8 * ch, depth=3, channels=8 * ch),
                                   DeconvUpsamplingBlock2D(in_channel=8 * ch, depth=3, channels=4 * ch),
                                   DeconvUpsamplingBlock2D(in_channel=4 * ch, depth=3, channels=2 * ch),
                                   DeconvUpsamplingBlock2D(in_channel=2 * ch, depth=3, channels=1 * ch),
                                   DeconvUpsamplingBlock2D(in_channel=1 * ch, depth=3, channels=1 * ch)])

        self.colorize = nn.Conv2d(ch, 3, kernel_size=(3, 3), padding=1)
        pass

    def forward(self, z, class_id):
        class_emb = self.class_emb(class_id)
        y = torch.cat((z, class_emb), dim=1)

        z_list = []
        if self.cell_mode == 'lstm':
            init_state = (y, y)
        elif self.cell_mode == 'gru':
            init_state = y
        else:
            raise NotImplementedError('the {} should be defined'.format(self.cell_mode))

        state = init_state
        z = init_state[0] if isinstance(init_state, tuple) else init_state

        for i in range(self.n_frames):
            state = self.initial_generator(z, state)
            z_list.append(state[0] if isinstance(state, tuple) else state)

        z = torch.cat(z_list, dim=0)
        z = self.transform(z)
        out = z.view(-1, 8 * self.channel, self.latent_dim, self.latent_dim).contiguous()  # BT, C, H, W

        layer_out_list = []
        for conv in self.conv:
            out = conv(out)
            _, c, h, w = out.size()
            layer_out_list.append(out.view(-1, self.n_frames, c, h, w).contiguous()[:, :, 0, :, :].squeeze())
        out = self.colorize(out)

        BT, C, H, W = out.size()
        out = out.view(-1, self.n_frames, C, H, W)
        return out, layer_out_list


if __name__ == '__main__':
    z = torch.randn(1, 120)
    id = torch.randint(0, 101, (1,))

    G = SimpleG()
    y = G(z, id)
    print(y[0].size())
