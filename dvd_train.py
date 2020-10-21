# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:26:00 2020

@author: Antony
"""

import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn

from model.dvd_modules.Generator import Generator
from model.dvd_modules.Discriminator import SpatialDiscriminator, TemporalDiscriminator

from model.simple_modules.generator import SimpleG
from model.simple_modules.discriminator import SimpleSD, SimpleTD

from dataloader import get_dataloader
from model.util import get_scheduler

from util.visualizer import Visualizer

from model.dvd_gan import DVDGAN


def main():
    parser = argparse.ArgumentParser(description='Video Self-disentanglement, PyTorch Version')
    # basic config
    parser.add_argument('--frames_saved_path', type=str, default='data/101frames')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--split_type', type=str, default='01')
    parser.add_argument('--preproc', type=int, default=1)

    parser.add_argument('--model_save_path', type=str, default='saved_model')
    parser.add_argument('--model_save_date', type=str, required=True)  # e.g. 'July29'
    parser.add_argument('--is_variation', type=bool, default=False)
    parser.add_argument('--net_type', type=str, default='gru', help='gru or 3d')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--name', type=str, default='',
                        help='name of the experiment. It decides where to store samples and models')

    # training config
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--net_init_type', type=str, default='normal', help='kaiming, orthogonal or xavier')

    parser.add_argument('--gan_mode', type=str, default='wgan-gp')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--distributed', type=bool, default=True)
    parser.add_argument('--optimizer_mode', type=str, default='Adam')
    parser.add_argument('--n_class', type=int, default=101)
    parser.add_argument('--z_dim', type=int, default=120)

    parser.add_argument('--g_lr', type=float, default=5e-5)
    parser.add_argument('--ds_lr', type=float, default=5e-5)
    parser.add_argument('--dt_lr', type=float, default=5e-5)

    parser.add_argument('--grad_accumulation', type=str, default=False)
    parser.add_argument('--lr_reduce_mode', type=str, default='step', help='step or adaptive')
    parser.add_argument('--lr_reduce_epoch', type=int, default=1, help='if args.ReduceLROnPlateau, this mean patience')
    parser.add_argument('--lr_reduce_ratio', type=float, default=0.9)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--test_interval', type=int, default=20)
    parser.add_argument('--save_model_interval', type=int, default=1)

    # visualization config
    parser.add_argument('--visualized', type=bool, default=False)
    parser.add_argument('--display_id', type=int, default=1)
    parser.add_argument('--display_ncols', type=int, default=16,
                        help='if positive, display all images in a single visdom web panel with certain number of '
                             'images per row.')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom display port')
    parser.add_argument('--display_env', type=str, default='main',
                        help='visdom display environment name (default is "main")')
    parser.add_argument('--display_server', type=str, default="http://10.198.8.31",
                        help='visdom server of the web display')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')

    # loss weight
    parser.add_argument('--content_lambda', type=float, default=1.0, help='weight of gap_recon_loss')
    parser.add_argument('--kl_lambda', type=float, default=0.1, help='weight of reg_loss')

    parser.add_argument('--simple', type=bool, default=False, help='step or adaptive')

    args = parser.parse_args()
    print(str(args), '\n')

    if args.simple:
        G = SimpleG()
        SD = SimpleSD()
        TD = SimpleTD()
    else:
        G = Generator(n_class=101, n_frames=16, hierar_flag=False)
        SD = SpatialDiscriminator(chn=32, n_class=101)
        TD = TemporalDiscriminator(chn=32, n_class=101)

    net = DVDGAN(args, G, SD, TD)
    train_op(net, args)


def set_scheduler(optimizers, args):
    schedulers = [get_scheduler(optimizer, args) for optimizer in optimizers]
    return schedulers


def update_optimizer(optimizer, args, epoch):
    try:
        lr = optimizer.param_groups[0]['lr']
        if args.lr_reduce_mode == 'step' and (epoch + 1) % args.lr_reduce_epoch == 0:
            optimizer.param_groups[0]['lr'] = args.lr_reduce_ratio * lr
        # elif args.lr_reduce_mode == 'adaptive' and epoch - args.lr_reduce_epoch > adapt_epoch:
        #     optimizer.param_groups[0]['lr'] = args.lr_reduce_ratio * lr
        #     adapt_epoch = epoch
        elif args.lr_reduce_mode == 'milestone':
            ms_list = [20, 25, 30, 35, 40, 45, 50]
            if (epoch + 1) in ms_list:
                optimizer.param_groups[0]['lr'] = args.lr_reduce_ratio * lr
    except NameError('{} is not defined !'.format(optimizer)):
        pass


def sample_k_frames(data, video_length, k_sample):
    frame_idx = torch.randperm(video_length)
    srt, idx = frame_idx[:k_sample].sort()
    return data[:, srt, :, :, :]


def vid_downsample(data):
    out = data
    B, T, C, H, W = out.size()
    x = F.avg_pool2d(out.view(B * T, C, H, W), kernel_size=2)
    _, _, H, W = x.size()
    x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
    return x


def calc_d_loss(x, real_flag, mode):
    if real_flag is True:
        x = -x
    if mode == 'wgan':
        loss = x.mean()
    elif mode == 'hinge':
        loss = torch.nn.ReLU()(1.0 + x).mean()
    else:
        raise NotImplementedError
    return loss


def calc_g_loss(x):
    x = -x
    loss = x.mean()
    return loss


def get_current_loss(g_loss, ds_loss, dt_loss):
    loss_dict = OrderedDict()
    loss_dict['g_loss'] = float(g_loss)
    loss_dict['ds_loss'] = float(ds_loss)
    loss_dict['dt_loss'] = float(dt_loss)

    return loss_dict


def train_op(net, args):
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    net.net_init(device, list(range(torch.cuda.device_count())), args.distributed)
    # accelerate training
    cudnn.benchmark = True

    # define visualizer
    if args.visualized:
        visualizer = Visualizer(args)

    # params_dict = {}
    # for name, params in net.named_parameters():
    #     print(name, ':', params.size())
    #     params_dict[name] = params.detach().cpu().numpy()
    # print(net)

    train_loader, size = get_dataloader(args, seed=1227, need_label=True, preproc=args.preproc)

    # define optimizers
    optimizers = []

    optimizer_g = torch.optim.RMSprop(net.G.parameters(), lr=args.g_lr)
    optimizer_ds = torch.optim.RMSprop(net.SD.parameters(), lr=args.ds_lr)
    optimizer_dt = torch.optim.RMSprop(net.TD.parameters(), lr=args.dt_lr)
    optimizers.append(optimizer_g)
    optimizers.append(optimizer_ds)
    optimizers.append(optimizer_dt)

    for epoch in range(args.num_epochs):
        start_time = time.clock()
        print('training epoch starts at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        for batch_ind, (real, real_label) in enumerate(train_loader):

            real, real_label = Variable(real), Variable(real_label)
            real, real_label = real.to(device), real_label.to(device)

            torch.set_grad_enabled(True)
            ### =================== training procedure ================ ###
            ## sample z
            z = torch.randn(args.batch_size, args.z_dim).to(device)
            z_class = torch.randint(low=0, high=args.n_class, size=(args.batch_size,)).to(device)

            real, fake = net.forward(real, real_label, z, z_class)

            net.set_requires_grad([net.SD, net.TD], True)
            ## =========== DS ============= ##
            ds_real = net.SD(real, real_label)
            ds_fake = net.SD(fake, z_class)

            ds_real_loss = calc_d_loss(ds_real, True, args.gan_mode)
            ds_fake_loss = calc_d_loss(ds_fake.detach(), False, args.gan_mode)
            ds_loss = ds_real_loss + ds_fake_loss

            optimizer_ds.zero_grad()
            ds_loss.backward()
            optimizer_ds.step()

            ## =========== DT ============= ##
            downsample_real = vid_downsample(real)
            downsample_fake = vid_downsample(fake)
            dt_real = net.TD(downsample_real, real_label)
            dt_fake = net.TD(downsample_fake, z_class)

            dt_real_loss = calc_d_loss(dt_real, True, args.gan_mode)
            dt_fake_loss = calc_d_loss(dt_fake.detach(), False, args.gan_mode)
            dt_loss = dt_real_loss + dt_fake_loss

            optimizer_dt.zero_grad()
            dt_loss.backward()
            optimizer_dt.step()

            net.set_requires_grad([net.SD, net.TD], False)
            ## =========== G ============= ##
            g_s_fake = net.SD(fake, z_class)
            g_t_fake = net.TD(downsample_fake, z_class)
            g_s_loss = calc_g_loss(g_s_fake)
            g_t_loss = calc_g_loss(g_t_fake)
            g_loss = g_s_loss + g_t_loss

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            ### ================= display loss ================ ###
            ds_loss = ds_loss.cpu() if use_gpu else ds_loss
            dt_loss = dt_loss.cpu() if use_gpu else dt_loss
            g_loss = g_loss.cpu() if use_gpu else g_loss

            if batch_ind % (len(train_loader) // 5) == 0:
                print('=' * 200)
                print('epoch {} --- iteration {}: '
                      'g loss = {:.6f} '
                      'ds loss = {:.6f} '
                      'dt loss = {:.6f} '.format(epoch, batch_ind, g_loss, ds_loss, dt_loss))

            if args.visualized:
                ### ===================== plot losses ==================== ###
                visualizer.reset()
                visualizer.plot_current_losses(epoch, float(batch_ind * args.batch_size) / size,
                                               get_current_loss(g_loss, ds_loss, dt_loss))
                net.obtain_var_names()
                visualizer.plot_current_variable_norm(
                    epoch, float(batch_ind * args.batch_size) / size, net.get_current_norm_visual())

            if args.visualized:
                ### ===================== visualization ==================== ###
                # visualize images
                visualizer.reset()
                net.obtain_frames_names()
                visualizer.display_current_results(
                    net.get_current_frame_visual(), epoch, True, preproc_mode=args.preproc, separate=False)
                visualizer.display_current_results(
                    net.get_layer_visual(), epoch, True, preproc_mode=args.preproc)
                visualizer.plot_current_layers_norm(
                    epoch, float(batch_ind * args.batch_size) / size, net.get_layer_visual(), idx=90)


        epoch_training_time = time.clock() - start_time
        print('training epoch ends at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        print('epoch: {}, epoch training time: {:.2f} min\n'.format(epoch, epoch_training_time / 60))

        # model test
        if (epoch + 1) % args.test_interval == 0:
            # TODO: TEST
            pass

        # lr scheduler
        for optimizer in optimizers:
            update_optimizer(optimizer, args, epoch)


if __name__ == '__main__':
    main()
    # net = ContentEncoder2()
    # print(net)
