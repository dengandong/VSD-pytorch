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

from collections import OrderedDict

from dataloader import get_dataloader
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
        from model.autoencoder.content_encoder import ContentEncoder2
        dCE = ContentEncoder2()
    elif args.net_type == '3d':
        from model.autoencoder.ae_3dcnn import ContentEncoder
        dCE = ContentEncoder()
        # cD = CodeDiscriminator()
    elif args.net_type == '2d':
        from model.autoencoder.ae_2dcnn import ContentEncoder
        dCE = ContentEncoder()
    else:
        raise NotImplementedError('net type has not been correctly claimed')
    net = dCE
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


def compute_loss(x, y):
    l2 = torch.nn.MSELoss()
    loss = l2(x, y)
    return loss


def get_current_image_visual(image1, image2):
    image_dict = OrderedDict()
    image_dict['image1'] = image1
    image_dict['image2'] = image2

    return image_dict


def get_current_code_visual(code):
    image_dict = OrderedDict()
    image_dict['image'] = code
    image_dict['code'] = code
    return image_dict


def get_current_loss(loss):
    loss_dict = OrderedDict()
    loss_dict['loss'] = float(loss)
    loss_dict['loss2'] = float(loss)
    return loss_dict


def train_op(net, args):
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
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

    train_img_loader = get_dataloader(args)

    # define optimizers
    optimizers = []

    optimizer_dce = torch.optim.Adam(net.parameters(), lr=args.dce_lr)
    optimizers.append(optimizer_dce)

    for epoch in range(args.num_epochs):
        start_time = time.clock()
        print('training epoch starts at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        for batch_ind, data1 in enumerate(train_img_loader):

            data1 = data1[:, 0]
            data1 = Variable(data1)
            data1 = data1.to(device)

            torch.set_grad_enabled(True)
            ### =================== training procedure ================ ###

            code, _x, _y = net.forward(data1)
            loss = compute_loss(_x, _y)
            optimizer_dce.zero_grad()
            loss.backward()
            optimizer_dce.step()

            ### ================= display loss ================ ###
            recon_loss = loss.cpu() if use_gpu else loss
            if batch_ind % (len(train_img_loader) // 5) == 0:
                print('=' * 200)
                print('epoch {} --- iteration {}: '
                      'reconstruction loss = {:.6f} '.format(epoch, batch_ind,
                                                                  recon_loss))

            if args.visualized:
                ### ===================== plot losses ==================== ###
                visualizer.reset()
                visualizer.plot_current_losses(epoch, float(batch_ind) / len(train_img_loader),
                                               get_current_loss(recon_loss))
                visualizer.plot_current_variable_norm(epoch, float(batch_ind / len(train_img_loader)),
                                                      get_current_code_visual(code), idx=10)



            if args.visualized:
                ### ===================== visualization ==================== ###
                # visualize images
                # visualizer.reset()
                visualizer.display_current_results(get_current_image_visual(_x, _y), epoch, True, 2, 'images')

        epoch_training_time = time.clock() - start_time
        print('training epoch ends at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        print('epoch: {}, epoch training time: {:.2f} min\n'.format(epoch, epoch_training_time / 60))

        # model test
        if (epoch+1) % args.test_interval == 0:
            # TODO: TEST
            pass

        # lr scheduler
        update_optimizer(optimizer_dce, args, epoch)


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


