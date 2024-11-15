import os
import cv2
import numpy as np
import pickle
from sklearn.utils import shuffle
from PIL import Image
import torch
import torch.nn.functional as F
import os
import numpy as np
import cv2
import torchvision.transforms as transforms
import itertools
import random
from scipy import io
from sklearn.utils import shuffle
from utility import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvideotransforms import video_transforms, volume_transforms


def load_video(path, label, vid_sample_len=64, stage='train', args=None):
    cap = cv2.VideoCapture(path)

    if stage =='test' and 'QUVA' in args.test_DB:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        num_frames = label.shape[0]

    max_frame_len = 300 #equal length as countix
    start_f = 0
    stride = 1

    if stage == 'train':
        width = 256  # 256
        height = 256  # 256

        strides = [1, 2, 4]

        if num_frames > max_frame_len:

            if 'train' in args.stride_sampling:
                max_stride = int(num_frames/max_frame_len)
                s = random.randint(0, 2)
                stride = strides[s]

                while stride > max_stride:
                    s -= 1
                    stride = strides[s]
            else:
                stride = 1

            start_f = random.randint(0, num_frames - max_frame_len * stride)
        else:
            max_frame_len = num_frames

    else:
        width = 224
        height = 224
        max_frame_len = num_frames


    if stage == 'train' and (args.model_choice == 'TransRac' or args.model_choice == 'RepNet'):
        taken = np.linspace(0, num_frames-1, vid_sample_len).astype(int)
    else:
        #taken = np.linspace(0, num_frames-1, num_frames).astype(int)
        taken = np.linspace(start_f, start_f+max_frame_len*stride-1, max_frame_len).astype(int)

    video = []
    label_new = []

    for fr_idx in range(num_frames):
        ret, frame = cap.read()

        if not ret: #중간에 가끔 프레임 없는 경우 예외처리
            # frame = np.zeros((width, height, 3))
            # video.append(frame.astype(np.float32))
            continue

        if cap.isOpened():
            for i in range(taken.size):
                if fr_idx == taken[i]:

                    frame = cv2.resize(frame, dsize=(width, height))
                    video.append(frame)
                    label_new.append(label[fr_idx])
                else:
                    continue

    # list to array
    video = np.array(video).astype(np.float32)
    label_new = np.array(label_new)
    cap.release()

    ### visualize
    # print('stride:', stride)
    # np_perframe_period = np.zeros(max_frame_len)
    # for i in range(label_new.shape[0]):
    #     np_perframe_period[i] = (label_new[i] == 1)
    #
    # color_bar = np.full((20, max_frame_len, 3), (0, 0, 255), dtype=np.uint8)
    # color_bar[:, np.where(np_perframe_period == 1), 0] = 0
    # color_bar[:, np.where(np_perframe_period == 1), 1] = 255
    # color_bar[:, np.where(np_perframe_period == 1), 2] = 0
    #
    # for fr_idx in range(max_frame_len):
    #     frame = video[fr_idx]
    #     frame = frame.astype(np.uint8)
    #
    #     color_bar_copy = color_bar.copy()
    #     color_bar_copy[:, fr_idx, 0] = 255
    #     color_bar_copy[:, fr_idx, 1] = 0
    #     color_bar_copy[:, fr_idx, 2] = 0
    #
    #     frame = cv2.putText(frame, str(np_perframe_period[fr_idx]), (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
    #                         2.0, (0, 0, 255), 3, cv2.LINE_AA)
    #
    #     cv2.imshow('frame', frame)
    #     cv2.imshow('color_bar', color_bar_copy)
    #     k = cv2.waitKey(33)
    #
    #     if k == 27:  # esc key
    #         break

    return video, label_new


def get_dataloaders(args):

    transformer_tra = transforms.Compose([video_transforms.RandomRotation(30),
                                          video_transforms.RandomCrop((224, 224)),
                                          video_transforms.RandomHorizontalFlip(),
                                          #video_transforms.CenterCrop((224,224)),
                                          volume_transforms.ClipToTensor(div_255=False)])

    transformer_val = transforms.Compose([volume_transforms.ClipToTensor(div_255=False)])
    # transformer_tra = transforms.Compose([ToTensor()]) # VisualCenterCrop(cropsize=(224, 224), central=True)
    # transformer_val = transforms.Compose([ToTensor()])

    dataset_training = RepeDetectDataset(transform=transformer_tra, stage='train', args=args)
    dataset_val      = RepeDetectDataset(transform=transformer_val, stage='val', args=args)
    dataset_test     = RepeDetectDataset(transform=transformer_val, stage='test', args=args)

    datasets = {'train': dataset_training, 'val': dataset_val, 'test': dataset_test}

    dataloaders = { 'train': DataLoader(datasets['train'], batch_size=args.batchsize, shuffle=True,
                                        num_workers=args.num_workers, drop_last=True, pin_memory=True),
                    'val': DataLoader(datasets['val'], batch_size=1, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False, pin_memory=True),
                    'test': DataLoader(datasets['test'], batch_size=1, shuffle=False, #batch_size=128
                                      num_workers=args.num_workers, drop_last=False, pin_memory=True),
                  }
    return dataloaders


class RepeDetectDataset(Dataset):

    def __init__(self,
                 transform=None,
                 stage='train',
                 args=None,
                 train_test_list=None):

        self.args = args
        self.transform = transform
        self.rgb_list = []
        self.labels = []
        self.stage = stage
        self.train_DB = args.train_DB
        self.test_DB = args.test_DB
        self.root_dir = args.root_dir

        if stage == 'train' or stage == 'val': # consider joint learning of multiple DB
            for i in range(len(self.train_DB)):
                if self.train_DB[i] == 'countix':
                    img_dir = os.path.join(self.root_dir, 'RBC/countix_{0}'.format(stage))
                    gt_dir = os.path.join(self.root_dir, 'RBC/countix_{0}_per_frame_periodicity'.format(stage))

        elif stage == 'test': ## test a single DB
            if 'countix' in self.test_DB:
                img_dir = os.path.join(self.root_dir, 'RBC/countix_{0}'.format(stage))
                gt_dir = os.path.join(self.root_dir, 'RBC/countix_{0}_per_frame_periodicity'.format(stage))
            elif 'PERTUBE' in self.test_DB:
                img_dir = os.path.join(self.root_dir, 'PERTUBE/videos')
                gt_dir = os.path.join(self.root_dir, 'PERTUBE/matrices')

        self.rgb_list += [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if
                          f.split(".")[-1] == "mp4" or f.split(".")[-1] == "avi" or f.split(".")[-1] == "npy"]
        self.labels += [os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir)) if
                        f.split(".")[-1] == "npy" or f.split(".")[-1] == "mat"]

        if stage == 'train':
            self.rgb_list, self.labels = shuffle(self.rgb_list, self.labels)

        if 'QUVA' in self.test_DB:
            self.rgb_list = self.rgb_list[1:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        rgbpath = self.rgb_list[idx]
        label_path = self.labels[idx]
        path_tok = label_path.split('.')

        if path_tok[-1] == 'mat':
            mat_file = io.loadmat(label_path)
            label = mat_file['PeriodicFrames']
            label = label[0]
        elif path_tok[-1] == 'npy':
            label = np.load(label_path)

        video, label = load_video(rgbpath, label, self.args.vid_sample_len, self.stage, self.args)
        video = self.video_transform(video)
        #sample = {'rgb': video, 'label': label}

        if self.transform:
            video = self.transform(video)

        # permute
        video = video.permute(1, 2, 3, 0)

        # append rgb_path to sample dictionary
        sample = {'rgb': video, 'label': label, 'rgbpath': rgbpath}

        return sample

    def video_transform(self, np_clip): #from imagent

        # Div by 255
        np_clip /= 255.

        # Normalization
        np_clip -= np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # mean
        np_clip /= np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 3)  # std

        return np_clip

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb, gt = sample['rgb'], sample['label']
        return {'rgb': torch.from_numpy(rgb.astype(np.float32)),
                'label': torch.from_numpy(np.asarray(gt))}

class VisualCenterCrop(object):
    """ Return a normalized number of frames. """

    def __init__(self, cropsize=(224, 224), central=True):
        self.cropsize = cropsize
        self.central = central

    def __call__(self, sample):
        rgb,  label = sample['rgb'], sample['label']
        if rgb.shape[0] != 1 and self.central:
            num_frame_rgb = len(rgb)
            orig_x = rgb.shape[1]
            orig_y = rgb.shape[2]
            startx = orig_x // 2 - (self.cropsize[0] // 2)
            starty = orig_y // 2 - (self.cropsize[1] // 2)
            rgb = rgb[:, starty:starty + self.cropsize[1], startx:startx + self.cropsize[0], :]
        else:
            raise ValueError("Not implemented")

        return {'rgb': rgb,
                'label': label}