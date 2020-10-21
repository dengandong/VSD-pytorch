import os
import torch
import argparse
import cv2 as cv
from torch.autograd import Variable

from model.model_3dcnn import Model3D as model
from model.autoencoder.ae_3dcnn import MotionEncoderC3D, ContentEncoder, Decoder
from model.discriminator import Discriminator
from dataloader import get_dataloader


def test(args):
    ME = MotionEncoderC3D()
    CE = ContentEncoder()
    G = Decoder()
    D = Discriminator()
    net = model(args, ME, CE, G, D)
    params = torch.load(args.model_file)
    net.load_state_dict(params['net'])

    net.eval()
    torch.set_grad_enabled(False)

    test_loader = get_dataloader(args, 'test')
    print('numbers of training set: {}'.format(len(test_loader) * args.batch_size))

    for index, test_data in enumerate(test_loader):
        test_data = Variable(test_data)
        net(test_data)
        reconstruction = net.reconstructed_frames.numpy()
        reconstruction *= 255.0
        print('testing the No.{} frames'.format(index))
        for i in range(16):
            img = reconstruction[:, i]
            img_path = os.path.join(args.img_save_path, str(index))
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            cv.imwrite(os.path.join(img_path, str(i) + '.jpg'), img)


def main():
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument('--frames_saved_path', type=str, default='data/101frames_16')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--split_type', type=str, default='01')
    parser.add_argument('--model_save_path', type=str, default='saved_model')
    parser.add_argument('--img_save_path', type=str, default='test_results')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    print(str(args), '\n')

    test(args)


if __name__ == '__main__':
    main()
