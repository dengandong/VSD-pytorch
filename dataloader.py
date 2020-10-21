import os
import math
import argparse
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


class DataFromFile:
    def __init__(self, args):
        self.frames_saved_path = args.frames_saved_path
        self.mode = args.mode

        self.sample_frequency = 1
        self.n_frames = 16

        if self.mode == 'train':
            self.video_name_path = 'data/ucfTrainTestlist/trainlist{}.txt'.format(args.split_type)
            self.num_epochs = args.num_epochs
            # self.split_ratio = 0.9
        elif self.mode == 'test':
            self.video_name_path = 'data/ucfTrainTestlist/test_list{}.txt'.format(args.split_type)
            self.num_epochs = 1

        self.video_name_list = open(self.video_name_path, 'r').readlines()
        pass

    def get_data(self):
        """
        :return:
        frames_path_list: contains 16 frames' names for a single video
        label: corresponding video labels
        """
        data_dir = 'data/ucfTrainTestlist'
        class_txt_path = os.path.join(data_dir, 'classInd.txt')
        label_txt_list = open(class_txt_path, 'r').readlines()
        label_dict = dict()
        for label_name in label_txt_list:
            line = label_name.split()  # class_index class_name
            label_dict[line[1]] = int(line[0]) - 1  # {'class_name': class_index}, 1 ~ 101 --> 0 ~ 100

        video_name_list = self.video_name_list
        frames_path_list = []
        video_label_list = []
        for video_name in video_name_list:
            name = video_name.split()  # video_class/video_name class_index
            video_path_name = name[0]  # video_class/video_name
            video_class_name = video_path_name.split('/')[0]  # video_class

            frames_list = []  # to save frames' absolute path (from 101frames_16) for a single video

            cur_n_frames = 0  # cur_n_frames <= self.n_frames
            cur_dir_list = os.listdir(os.path.join(self.frames_saved_path, video_path_name))
            n_available_frames = len(cur_dir_list) // self.sample_frequency  # total_number_frames // sample_freq

            assert n_available_frames >= self.n_frames, 'Not enough available frames, reduce sample frequency !'

            # while cur_n_frames <= self.n_frames:
            if len(cur_dir_list) - self.n_frames * self.sample_frequency == 0:
                start_i = 0
            else:
                start_i = np.random.randint(0, len(cur_dir_list) - self.n_frames * self.sample_frequency)
            for i, frame_name in enumerate(cur_dir_list):
                if (i - start_i) % self.sample_frequency == 0:
                    frames_list.append(os.path.join(self.frames_saved_path, video_path_name, frame_name))
                    cur_n_frames += 1
                if cur_n_frames >= self.n_frames:
                    break

            frames_list.sort()
            # the element in frames_path_list is frames_list which contains 16 frames' name
            frames_path_list.append(frames_list)
            video_label_list.append(video_class_name)

        numeral_label_list = []
        for class_name in video_label_list:
            numeral_label_list.append(label_dict[class_name])
        label = np.array(numeral_label_list)
        return frames_path_list, label


class VideoPipeline(Dataset):
    def __init__(self, frames_path_list, label, need_label, transform=None, preproc=1):
        super(VideoPipeline, self).__init__()

        self.label = label
        self.transform = transform
        self.image_path_list = frames_path_list
        self.preproc = preproc
        self.need_label = need_label

    def __getitem__(self, index):
        video_label = self.label[index]
        video_frame = self.image_name_list_to_3d(self.image_path_list[index])
        cropped_video_frame = self.random_crop(video_frame)
        standard_video_frame = self.standardization(cropped_video_frame, mode=self.preproc)
        if self.transform:
            video_as_tensor = self.transform(standard_video_frame)
        else:
            video_as_tensor = torch.from_numpy(standard_video_frame)
            video_as_tensor = video_as_tensor.type(torch.FloatTensor)

        if self.need_label:
            return video_as_tensor, video_label
        else:
            return video_as_tensor

    def __len__(self):
        return len(self.label)

    def image_name_list_to_3d(self, img_name_list):
        img_list = []
        for image_name in img_name_list:
            img = cv.imread(image_name)
            img = np.array(img)  # (320, 240, 3)
            img = self.resize(img)  # (171, 128, 3)
            img = np.transpose(img, (2, 0, 1))
            img_list.append(img)
        img_3d = np.stack(img_list, axis=0)  # [16, 3, 171, 128]
        # assert np.shape(img_3d)[1] == 16, print('img_3d got wrong shape:', np.shape(img_3d))
        return img_3d
    
    @staticmethod
    def center_crop(img, h=128, w=128):
        _, _, img_h, img_w = np.shape(img)
        crop_img = img[:, :, img_h // 2 - h // 2:img_h // 2 - h // 2 + h, img_w // 2 - w // 2:img_w // 2 - w // 2 + w]
        return crop_img
    
    @staticmethod
    def random_crop(img, h=128, w=128):
        _, _, img_h, img_w = np.shape(img)
        if img_h > h and img_w > w:
            start_h = np.random.randint(low=0, high=img_h - h)
            start_w = np.random.randint(low=0, high=img_w - w)
            crop_img = img[:, :, start_h: start_h + h, start_w: start_w + w]
        elif img_h == h and img_w > w:
            start_w = np.random.randint(low=0, high=img_w - w)
            crop_img = img[:, :, :, start_w: start_w + w]
        elif img_h > h and img_w == w:
            start_h = np.random.randint(low=0, high=img_h - h)
            crop_img = img[:, :, start_h: start_h + h, :]
        else:
            crop_img = img
        return crop_img
    
    @staticmethod
    def resize(img, short=128):
        assert img.ndim == 3, print('wrong img shape', np.shape(img))
        img_h, img_w, _ = np.shape(img)
        ref_side = min(img_h, img_w)
        ratio = short / ref_side
        img = cv.resize(img, (math.ceil(img_w * ratio), math.ceil(img_h * ratio)))
        return img
    
    @staticmethod
    def standardization(img_3d, mode=1):
        if mode == 0:
            std_img = img_3d / 255.0
        elif mode == 1:
            std_img = (img_3d / 255.0) * 2 - 1  # scale to (-1, 1)
        else:
            raise NotImplementedError('{} is not implemented'.format(mode))
        return std_img


def get_dataloader(args, need_label=False, seed=None, preproc=1):

    frame_path_list, label = DataFromFile(args).get_data()
    dataset = VideoPipeline(frame_path_list, label, need_label=need_label, preproc=preproc)
    size = len(dataset)
    index = list(range(size))
    if seed is not None:
        np.random.seed(seed)
    else:
        seed = np.random.randint(low=0, high=9999)
        np.random.seed(seed)
    index = np.random.permutation(index)
    
    sampler = SubsetRandomSampler(index)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler, 
                            num_workers=args.num_workers, pin_memory=True)
    return dataloader, size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_saved_path', type=str, default='data/101frames_16')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--split_type', type=str, default='01')
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()

    dataloader = get_dataloader(args)
    print(len(dataloader))
