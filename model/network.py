import argparse
import numpy as np
import torch
import torch.nn as nn
from model import time_axis_expand
from collections import OrderedDict
from .util import init_net
from torch.autograd import Variable

from model.discriminator import Discriminator


class Model3D:
    name = 'model_3dcnn'

    def __init__(self,
                 args,
                 motion_encoder,
                 d_content_encoder,
                 v_content_encoder,
                 generator,
                 frame_discriminator,
                 ):
        # net initialization
        self.init_type = args.net_init_type

        # module definition
        self.ME = motion_encoder
        self.dCE = d_content_encoder  # deterministic, explicitly extract content
        self.vCE = v_content_encoder  # variational
        self.G = generator
        self.fD = frame_discriminator
        # self.iD = image_discriminator
        # self.cD = code_discriminator

        # criterion
        self.compute_mse = nn.MSELoss()
        self.compute_mae = nn.L1Loss()
        self.compute_bce = nn.BCELoss()

        self.gan_mode = args.gan_mode
        self.var_list = []

        self.loss = dict()

        # loss to be ploted
        self.stg1_loss_name = ['recon', 'gap_recon']
        self.stg2_loss_name = ['reg', 'img_recon', 'd', 'g', 'code']

    def net_init(self, device, gpu_ids):
        self.ME = init_net(self.ME, device, gpu_ids, init_type=self.init_type)
        self.dCE = init_net(self.dCE, device, gpu_ids, init_type=self.init_type)
        self.vCE = init_net(self.vCE, device, gpu_ids, init_type=self.init_type)
        self.G = init_net(self.G, device, gpu_ids, init_type=self.init_type)
        self.fD = init_net(self.fD, device, gpu_ids, init_type=self.init_type)
        # self.iD = init_net(self.iD, device, gpu_ids)
        # self.cD = init_net(self.cD, device, gpu_ids)
        pass

    def deterministic_forward(self, input_frames):

        self.true = input_frames  # B x T x C x H x W
        self.z_m = self.ME(self.true)
        ## deterministic part
        self.z_c, self.gap, self.reconstructed_gap = self.dCE(self.true)  # ae
        self.z_d = self.z_m + self.z_c  # B x 120
        self.reconstructed_frames = self.G(self.z_d)
        self.incompleted_frames = self.G(self.z_m)  # only decode the z_m
        self.diff_m_d = self.reconstructed_frames - self.incompleted_frames
        self.static_frames = self.G(self.z_c)  # only decode the z_c

        # def print_grad(grad):
        #     print(grad)
        # print('='*50, 'grad of the gap', '='*50)
        # self.reconstructed_gap.register_hook(print_grad)

        ## ============== calculate loss ============== ##
        loss = self.loss
        self.recon_loss = torch.mean(self.compute_mse(self.true, self.reconstructed_frames))
        self.gap_recon_loss = torch.mean(self.compute_mse(self.gap, self.reconstructed_gap))
        loss['recon'] = self.recon_loss
        loss['gap_recon'] = self.gap_recon_loss
        return loss

    def variational_forward(self, input_frames, input_imgs, eps):

        self.true = input_frames  # B x T x C x H x W
        self.z_m = self.ME(self.true)
        ## variational part
        self.img = input_imgs
        self.z_c_mu, self.z_c_log_sigma_square, self.reconstructed_img = self.vCE(self.img, eps)  # vae
        eps = time_axis_expand(eps)
        self.z_c1 = self.z_c_mu + torch.exp(self.z_c_log_sigma_square / 2) * eps
        self.z_v = self.z_m + self.z_c1
        self.synthesized_frames = self.G(self.z_c1)
        self.var_list.append(0)
        ## obtain motion code again
        self.z_m_1 = self.ME(self.synthesized_frames)

        ## ============== calculate loss ============== ##
        loss = self.loss
        # kl divergence
        self.reg_loss = self.compute_regularization_loss()
        loss['reg'] = self.reg_loss
        # img reconstruction loss
        self.img_recon_loss = torch.mean(self.compute_mae(self.img, self.reconstructed_img))
        loss['img_recon'] = self.img_recon_loss
        # gan loss
        self.d_loss, self.g_loss = self.compute_adversarial_loss(self.fD, self.true, self.synthesized_frames)
        loss['d'] = self.d_loss  # frame d_loss
        loss['g'] = self.g_loss
        # code reconstruction loss and code adversarial loss
        self.code_loss = self.compute_representation_loss()
        loss['code'] = self.code_loss
        # self.code_d_loss, self.code_g_loss = self.compute_adversarial_loss(self.cD, self.z_m, self.z_m_1)
        # loss['code_d'] = self.code_d_loss
        # loss['code_g'] = self.code_g_loss
        return loss

    def compute_adversarial_loss(self, net, real, generated):
        """
        adversarial loss to produce realistic synthetic images after decoder
        """
        real_score = net(real)
        false_score = net(generated.detach())
        if self.gan_mode in ['bce', 'ls']:
            sigmoid = nn.Sigmoid()
            real_score, false_score = sigmoid(real_score), sigmoid(false_score)
            if self.gan_mode == 'bce':
                real_loss = self.compute_bce(real_score, torch.ones_like(real_score))
                false_loss = self.compute_bce(false_score, torch.zeros_like(false_score))
                g_loss = self.compute_bce(false_score, torch.ones_like(false_score))
            elif self.gan_mode == 'ls':
                real_loss = self.compute_mse(real_score, torch.ones_like(real_score))
                false_loss = self.compute_mse(false_score, torch.zeros_like(false_score))
                g_loss = self.compute_mse(false_score, torch.ones_like(false_score))
            else:
                raise NotImplementedError('{} not implemented'.format(self.gan_mode))
            d_loss = real_loss + false_loss

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

    def compute_code_d_loss(self, real_code, gen_code):
        real_code_score = self.cD(real_code)
        fake_code_score = self.cD(gen_code)
        real_loss = self.compute_bce(real_code_score, torch.ones_like(real_code_score))
        fake_loss = self.compute_bce(fake_code_score, torch.zeros_like(fake_code_score))

        code_d_loss = real_loss + fake_loss
        return code_d_loss

    def compute_regularization_loss(self):
        """
        regularization loss to force the posterior distribution closer to Gaussian, i.e., reduce the KL-divergence
        between the two. (VAE loss)
        """
        # kl_divergence = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1) ??
        kl_divergence = 0.5 * torch.sum(
            torch.exp(self.z_c_log_sigma_square) + torch.square(self.z_c_mu) - 2 * self.z_c_mu - self.z_c_log_sigma_square, 1)
        kl_loss = torch.mean(kl_divergence)
        return kl_loss

    def compute_representation_loss(self, mode='l2'):
        """
        representation loss to make all the synthetic code z_m_1 after encoder approximate the z_m_0, i.e.,
        force the encoder to filter all the low-level information
        """
        if mode == 'l2':
            return torch.mean(self.compute_mse(self.z_m, self.z_m_1))
        elif mode == 'l1':
            return torch.mean(self.compute_mae(self.z_m, self.z_m_1))

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_loss(self, is_stg1=False):
        loss_dict = OrderedDict()
        if is_stg1:
            for name in self.stg1_loss_name:
                if isinstance(name, str):
                    loss_dict[name] = float(getattr(self, name + '_loss'))
        else:
            for name in self.stg2_loss_name:
                if isinstance(name, str):
                    loss_dict[name] = float(getattr(self, name + '_loss'))
        return loss_dict

    def obtain_frames_names(self):
        # compute variables to be visualized after forward
        if self.var_list:
            self.frames_names = ['true', 'reconstructed_frames', 'synthesized_frames']
        else:
            self.frames_names = ['true', 'reconstructed_frames', 'incompleted_frames', 'static_frames']
        pass

    def get_current_frame_visual(self):
        visual_dict = OrderedDict()
        for name in self.frames_names:
            if isinstance(name, str):
                visual_dict[name] = getattr(self, name)
        return visual_dict

    def obtain_images_names(self):
        # compute variables to be visualized after forward
        if self.var_list:
            self.images_names = ['img', 'reconstructed_img']
        else:
            self.images_names = ['gap', 'reconstructed_gap']
        pass

    def get_current_image_visual(self):
        visual_dict = OrderedDict()
        for name in self.images_names:
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
        if self.var_list:
            self.var_names = ['z_c1', 'z_m', 'z_v']
        else:
            self.var_names = ['z_c', 'z_m', 'z_d', 'diff_m_d']

    def get_current_norm_visual(self):
        visual_dict = OrderedDict()
        for name in self.var_names:
            if isinstance(name, str):
                visual_dict[name] = getattr(self, name)
        return visual_dict


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

