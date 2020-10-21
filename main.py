# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:26:00 2020

@author: Antony
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn

from dataloader import get_dataloader
from model.network import Model3D as model
from model.discriminator import Discriminator
from model.util import get_scheduler

from util.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(description='Video Self-disentanglement, PyTorch Version')
    # basic config
    parser.add_argument('--frames_saved_path', type=str, default='data/101frames')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--split_type', type=str, default='01')

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

    parser.add_argument('--me_lr', type=float, default=1e-4)
    parser.add_argument('--dce_lr', type=float, default=1e-4)
    parser.add_argument('--vce_lr', type=float, default=1e-5)
    parser.add_argument('--g_lr', type=float, default=1e-5)
    parser.add_argument('--d_lr', type=float, default=1e-5)

    parser.add_argument('--grad_accumulation', type=str, default=False)
    parser.add_argument('--lr_reduce_mode', type=str, default='step', help='step or adaptive')
    parser.add_argument('--lr_reduce_epoch', type=int, default=1, help='if args.ReduceLROnPlateau, this mean patience')
    parser.add_argument('--lr_reduce_ratio', type=float, default=0.5)
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

    args = parser.parse_args()
    print(str(args), '\n')
    if args.net_type == 'gru':
        from model.autoencoder.ae_gru import MotionEncoderGRU, DecoderGRU
        from model.autoencoder.content_encoder import ContentEncoder2
        ME = MotionEncoderGRU()
        dCE = ContentEncoder2()
        vCE = ContentEncoder2()
        G = DecoderGRU()
        fD = Discriminator()
    elif args.net_type == '3d':
        from model.autoencoder.ae_3dcnn import MotionEncoderC3D, ContentEncoder, VariationalContentEncoder, Decoder
        ME = MotionEncoderC3D()
        dCE = ContentEncoder()
        vCE = VariationalContentEncoder()
        G = Decoder()
        fD = Discriminator()
        # cD = CodeDiscriminator()
    elif args.net_type == '2d':
        from model.autoencoder.ae_2dcnn import MotionEncoderC2D, ContentEncoder, VariationalContentEncoder, DecoderC2D
        ME = MotionEncoderC2D()
        dCE = ContentEncoder()
        vCE = VariationalContentEncoder()
        G = DecoderC2D()
        fD = Discriminator()
    else:
        raise NotImplementedError('net type has not been correctly claimed')
    net = model(args, ME, dCE, vCE, G, fD)
    train_op(net, args)


# def set_model(net, device, is_parallel=True):
#     net.to(device)
#     if is_parallel:
#         net = nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
#         print("Using", torch.cuda.device_count(), "GPUs")
#         # net = net.module
#     return net


def set_optimizer(optimizer, is_parallel=True):
    # if is_parallel:
    #     optimizer = nn.DataParallel(optimizer, device_ids=list(range(torch.cuda.device_count())))
    #     optimizer = optimizer.module
    return optimizer


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


def train_op(net, args):
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    net.net_init(device, list(range(torch.cuda.device_count())))
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

    train_loader, n_train_samples = get_dataloader(args, seed=1227, need_label=False)
    train_img_loader, _ = get_dataloader(args)
    print('numbers of training set: {}'.format(len(train_loader) * args.batch_size))

    # define optimizers
    optimizers = []
    optimizer_g = torch.optim.RMSprop(net.G.parameters(), lr=args.g_lr)
    optimizers.append(optimizer_g)
    optimizer_d = torch.optim.RMSprop(net.fD.parameters(), lr=args.d_lr)
    optimizers.append(optimizer_d)
    optimizer_me = torch.optim.Adam(net.ME.parameters(), lr=args.me_lr)
    optimizers.append(optimizer_me)
    optimizer_dce = torch.optim.Adam(net.dCE.parameters(), lr=args.dce_lr)
    optimizers.append(optimizer_dce)
    optimizer_vce = torch.optim.Adam(net.vCE.parameters(), lr=args.vce_lr)
    optimizers.append(optimizer_vce)

    for epoch in range(args.num_epochs):
        start_time = time.clock()
        print('training epoch starts at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        for batch_ind, (data, data1) in enumerate(zip(train_loader, train_img_loader)):

            data = Variable(data)
            data = data.to(device)

            data1 = data1[:, 0]
            data1 = Variable(data1)
            data1 = data1.to(device)

            # z_eps = np.random.normal(0, 1, (args.batch_size, 16, 256))
            z_eps = np.random.normal(0, 1, (args.batch_size, 256, 8, 8))
            z_eps = torch.Tensor(z_eps).to(device)

            torch.set_grad_enabled(True)
            ### =================== training procedure ================ ###

            if epoch <= 40:
                loss = net.deterministic_forward(data)

                ## update ME, dCE, G: deterministic stage
                optimizer_me.zero_grad()
                optimizer_dce.zero_grad()
                optimizer_g.zero_grad()

                recon_loss = loss['recon'].mean()
                gap_recon_loss = loss['gap_recon'].mean()
                ae_loss = recon_loss + args.content_lambda * gap_recon_loss
                ae_loss.backward()

                optimizer_me.step()
                optimizer_dce.step()
                optimizer_g.step()

                ### ================= display loss ================ ###
                recon_loss = recon_loss.cpu() if use_gpu else recon_loss
                gap_recon_loss = gap_recon_loss.cpu() if use_gpu else gap_recon_loss
                if batch_ind % (len(train_loader) // 5) == 0:
                    print('=' * 200)
                    print('epoch {} --- iteration {}: '
                          'reconstruction loss = {:.6f} '
                          'gap reconstruction loss = {:.6f}\n'.format(epoch, batch_ind,
                                                                      recon_loss, gap_recon_loss))

                if args.visualized:
                    ### ===================== plot losses ==================== ###
                    visualizer.reset()
                    visualizer.plot_current_losses(epoch, float(batch_ind / n_train_samples),
                                                   net.get_current_loss(is_stg1=True))
                    net.obtain_var_names()
                    visualizer.plot_current_variable_norm(epoch, float(batch_ind / n_train_samples),
                                                          net.get_current_norm_visual(), idx=10)

            else:
                loss = net.variational_forward(data, data1, z_eps)  # data: video; data1: image

                ## update ME, vCE, G: variational stage
                net.set_requires_grad(net.fD, False)
                optimizer_me.zero_grad()
                optimizer_vce.zero_grad()
                optimizer_g.zero_grad()

                # if lambda_kl >= 0.01:
                #     lambda_kl *= 0.9

                reg_loss = loss['reg'].mean()
                img_recon_loss = loss['img_recon'].mean()
                g_loss = loss['g'].mean()  # update G's params
                gan_vae_loss = g_loss + args.kl_lambda * reg_loss + img_recon_loss  # update vCE's params
                gan_vae_loss.backward(retain_graph=True)

                # fix ME and backward code loss in G
                net.set_requires_grad(net.ME, False)
                code_loss = loss['code'].mean()
                code_loss.backward(retain_graph=True)

                # code_g_loss = loss['code_g'].mean()
                # code_g_loss.backward(retain_graph=True)
                # # loose cD and update code_d
                # net.set_requires_grad(net.cD, True)
                # code_d_loss = loss['code_d'].mean()
                # code_d_loss.backward(retain_graph=True)

                net.set_requires_grad(net.ME, True)

                optimizer_vce.step()
                optimizer_me.step()
                optimizer_g.step()

                ## update d, frame_d
                net.set_requires_grad(net.fD, True)
                optimizer_d.zero_grad()
                d_loss = loss['d'].mean()  # some params have been updated?
                d_loss.backward()
                optimizer_d.step()

                ### =================== display loss ================ ###

                d_loss = d_loss.cpu() if use_gpu else d_loss
                # code_d_loss = code_d_loss.cpu() if use_gpu else d_loss
                g_loss = g_loss.cpu() if use_gpu else g_loss
                reg_loss = reg_loss.cpu() if use_gpu else reg_loss
                code_loss = code_loss.cpu() if use_gpu else code_loss
                img_recon_loss = img_recon_loss.cpu() if use_gpu else img_recon_loss

                if batch_ind % (len(train_loader) // 5) == 0:
                    print('=' * 200)
                    print('epoch {} --- iteration {}: '
                          'image reconstruction loss = {:.6f}\n'
                          'adversarial_d loss = {:.6f} '
                          'adversarial_g loss = {:.6f} '
                          'KL loss = {:.6f} '
                          'code z loss = {:.6f}'.format(epoch, batch_ind,
                                                        img_recon_loss, #code_d_loss,
                                                        d_loss, g_loss, reg_loss, code_loss))
                if args.visualized:
                    ### ===================== plot losses ==================== ###
                    print('plotting losses...')
                    visualizer.reset()
                    visualizer.plot_current_losses(epoch, float(batch_ind / n_train_samples),
                                                   net.get_current_loss(is_stg1=False))
                    net.obtain_var_names()
                    visualizer.plot_current_variable_norm(epoch, float(batch_ind / n_train_samples),
                                                          net.get_current_norm_visual(), idx=15)

            if args.visualized:
                ### ===================== visualization ==================== ###
                # visualize frames
                net.obtain_frames_names()
                visualizer.reset()
                visualizer.display_current_results(net.get_current_frame_visual(), epoch, True, 1, 'frames')
                # visualize images
                net.obtain_images_names()
                # visualizer.reset()
                visualizer.display_current_results(net.get_current_image_visual(), epoch, True, 2, 'images')
                # # visualize codes
                # net.obtain_codes_names()
                # # visualizer.reset()
                # visualizer.display_current_results(net.get_current_code_visual(), epoch, True, 3, 'codes')

        epoch_training_time = time.clock() - start_time
        print('training epoch ends at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        print('epoch: {}, epoch training time: {:.2f} min\n'.format(epoch, epoch_training_time / 60))

        # model test
        if (epoch+1) % args.test_interval == 0:
            # TODO: TEST
            pass

        # lr scheduler
        update_optimizer(optimizer_me, args, epoch)
        update_optimizer(optimizer_dce, args, epoch)
        update_optimizer(optimizer_g, args, epoch)
        update_optimizer(optimizer_d, args, epoch)
        update_optimizer(optimizer_vce, args, epoch)

        # # model saving
        # if (epoch+1) % args.save_model_interval == 0:
        #     state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        #     print('Saving model ...')
        #     model_saved_path = os.path.join(args.model_save_path, args.model_save_date)
        #     if not os.path.exists(model_saved_path):
        #         os.makedirs(model_saved_path)
        #     torch.save(state, os.path.join(model_saved_path, 'epoch_{}.pkl'.format(epoch)))
        #     print('Model was just saved in {}\n'.format(model_saved_path))


if __name__ == '__main__':
    main()
    # net = ContentEncoder2()
    # print(net)


