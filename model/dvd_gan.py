import argparse
import numpy as np
import torch
import torch.nn as nn
from model import time_axis_expand
from collections import OrderedDict
from model.util import init_net
from torch.autograd import Variable

from model.dvd_modules.Generator import Generator
from model.dvd_modules.Discriminator import SpatialDiscriminator, TemporalDiscriminator


def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


class DVDGAN:
    name = 'dvd-gan'

    def __init__(self, args, G, SD, TD):
        # net initialization
        self.init_type = args.net_init_type

        # criterion
        self.compute_bce = nn.BCELoss()

        self.gan_mode = args.gan_mode
        self.var_list = []

        self.loss = dict()

        # loss to be ploted
        self.loss_name = ['d_s', 'g', 'd_t']

        # module definition
        self.G = G
        self.SD = SD
        self.TD = TD
        pass

    def net_init(self, device, gpu_ids, distributed):
        self.G = init_net(self.G, device, gpu_ids, init_type=self.init_type, distributed=distributed)
        self.SD = init_net(self.SD, device, gpu_ids, init_type=self.init_type, distributed=distributed)
        self.TD = init_net(self.TD, device, gpu_ids, init_type=self.init_type, distributed=distributed)
        pass

    def forward(self, real, real_label, z, class_id):

        self.real = real
        self.fake, self.layer_output = self.G(z, class_id)

        fake = self.fake
        return real, fake

        # ## ============== calculate loss ============== ##
        # loss = self.loss
        #
        # self.d_s_loss, self.g_s_loss = self.compute_adversarial_loss(
        #     self.SD, self.real, self.fake, real_label, class_id)
        # self.d_t_loss, self.g_t_loss = self.compute_adversarial_loss(
        #     self.TD, self.real, self.fake, real_label, class_id)
        # self.g_loss = self.g_s_loss + self.g_t_loss
        #
        # loss['g'] = self.g_loss
        # loss['d_s'] = self.d_s_loss
        # loss['d_t'] = self.d_t_loss

        # return loss

    def compute_adversarial_loss(self, net, real, generated, real_label, class_id):
        """
        adversarial loss to produce realistic synthetic images after decoder
        """
        real_score = net(real, real_label)
        false_score = net(generated.detach(), class_id)

        if self.gan_mode == 'hinge':
            real_loss = -nn.ReLU()(1.0 + real_score)
            false_loss = nn.ReLU()(1.0 + false_score)
            d_loss = real_loss + false_loss
            g_loss = -nn.ReLU()(1.0 + false_score)

        else:  # wgan
            real_loss = -real_score.mean()
            false_loss = false_score.mean()
            g_loss = -false_score.mean()
            if self.gan_mode == 'wgan-clip':
                for p in net.parameters():
                    p.data.clamp_(-0.01, 0.01)
                d_loss = real_loss + false_loss

            elif self.gan_mode == 'wgan-gp':
                alpha = torch.rand(real.size(0), 1)
                # make alpha the same size as data
                alpha = alpha.expand(real.shape[0], real.nelement() // real.shape[0]).contiguous().view(*real.shape)
                alpha = alpha.to(torch.device('cuda:0'))
                print(real.size(), generated.size(), alpha.size())
                interpolate = real * alpha + generated * (1 - alpha)
                score = net(interpolate)
                gradients = torch.autograd.grad(outputs=score, inputs=interpolate,
                                                grad_outputs=torch.ones_like(score),
                                                create_graph=True, retain_graph=True, only_inputs=True)
                gradients = gradients[0].view(real.size(0), -1)  # flat the data
                gp = (((gradients + 1e-16).norm(2, dim=1) - 1.0) ** 2).mean()
                d_loss = real_loss + false_loss + 10 * gp
            elif self.gan_mode == 'wgan-vanilla':
                d_loss = real_loss + false_loss
            else:
                raise NotImplementedError('{} not implemented'.format(self.gan_mode))

        d_loss = torch.mean(d_loss)
        g_loss = torch.mean(g_loss)
        return d_loss, g_loss

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_loss(self):
        loss_dict = OrderedDict()
        for name in self.loss_name:
            if isinstance(name, str):
                loss_dict[name] = float(getattr(self, name + '_loss'))
        return loss_dict

    def obtain_frames_names(self):
        # compute variables to be visualized after forward
        self.frames_names = ['real', 'fake']

    def get_current_frame_visual(self):
        visual_dict = OrderedDict()
        for name in self.frames_names:
            if isinstance(name, str):
                visual_dict[name] = getattr(self, name)
        return visual_dict

    def obtain_codes_names(self):
        # compute variables to be visualized after forward
        if self.var_list:
            self.codes_names = ['z_m', 'z_v', 'z_m_1']
        else:
            self.codes_names = ['z_m', 'z_d']
        pass

    def get_current_code_visual(self):
        visual_dict = OrderedDict()
        for name in self.codes_names:
            if isinstance(name, str):
                visual_dict[name] = getattr(self, name)
        return visual_dict

    def obtain_var_names(self):
        self.var_names = ['real', 'fake']

    def get_current_norm_visual(self):
        visual_dict = OrderedDict()
        for name in self.var_names:
            if isinstance(name, str):
                visual_dict[name] = getattr(self, name)
        return visual_dict

    def get_layer_visual(self):
        layer_out_list = self.layer_output
        layer_dict = OrderedDict()
        for i, layer_out in enumerate(layer_out_list):
            # layer_out_0 = layer_out[0]
            # print(layer_out_0.size())  # (16, ld, ld)
            layer_dict['layer {}'.format(i)] = layer_out[0:]  # one of the batches
            # print(layer_out.size())
        return layer_dict


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batch_size', type=int, default=30)
#     args = parser.parse_args()
#
#     inputs = torch.randn(1, 3, 16, 64, 64)
#     ME = MotionEncoderC3D()
#     CE = ContentEncoder()
#     G = Decoder()
#     D = Discriminator()
#     model = Model3D(args, ME, CE, G, D)
#
#     model(inputs)
#     print('motion code:', model.z_m.size())
#     print('z_mu:', model.z_mu.size())
#     print('z_c:', model.z_c.size())
#     print('generated frames:', model.synthesized_frames.size())

