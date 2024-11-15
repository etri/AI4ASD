import torch
import torch.nn.functional as F
import os
import gc
import numpy as np
import random
import cv2
import torchvision.transforms as transforms
import itertools
import math
from sklearn.utils import shuffle
from PIL import Image
from scipy import io
from utility import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from datetime import timedelta
from torchvision.ops import nms


def seconds_to_timecode(seconds):
    # Create a timedelta object from seconds
    duration = timedelta(seconds=seconds)

    # Format the timedelta as HH:MM:SS.sss
    timecode = str(duration).split(".")[0]  # remove microseconds
    return timecode


def numpy_collate(batch):
    for item in batch:
        data_numpy = item

    return data_numpy


def get_dataloaders(args):
    transformer_val = transforms.Compose([ToTensor()])

    dataset_test = DATA(args.datadir, transform=transformer_val, stage='test', args=args)

    datasets = {'test': dataset_test}

    dataloaders = {
        'test': DataLoader(datasets['test'], batch_size=1, shuffle=False,
                           num_workers=args.num_workers,
                           drop_last=False, pin_memory=True, collate_fn=numpy_collate), ## feed data in numpy format, not tensor for memory efficiency
    }

    return dataloaders


class DATA(Dataset):

    def __init__(self, root_dir='',  # /home/juanma/Documents/Data/ROSE_Action
                 transform=None,
                 stage='train',
                 vid_len=(8, 32),
                 args=None):

        if args.data_choice == 'SSBD':

            action_cls = ['ArmFlapping', 'Spinning', 'HeadBanging']
            basename = os.path.join(root_dir, 'Dataset_revised_ych')

            self.input_list = []
            self.labels = []
            self.gt_list = []

            for i in range(len(action_cls)):

                basename_cls = os.path.join(basename,action_cls[i])
                input_list = []
                input_list += [os.path.join(basename_cls, f) for f in sorted(os.listdir(basename_cls)) if
                                            f.split(".")[-1] == "mp4" or f.split(".")[-1] == "avi"]
                self.input_list.append(input_list)
                labels = [i for j in range(len(self.input_list[i]))]
                self.labels.append(labels)

                gt_cls = os.path.join(basename, action_cls[i]) + '_per_frame_periodicity_colorful'
                self.gt_list += [os.path.join(gt_cls, f) for f in sorted(os.listdir(gt_cls)) if f.split(".")[-1] == "npy"]

            self.input_list = list(itertools.chain.from_iterable(self.input_list))
            self.labels = list(itertools.chain.from_iterable(self.labels))

        elif args.data_choice == 'ESBD':

            action_cls = ['ArmFlapping', 'Spinning', 'HeadBanging', 'HandAction']
            basename = os.path.join(root_dir, 'Dataset')

            self.input_list = []
            self.labels = []
            self.gt_list = []

            for i in range(len(action_cls)):
                basename_cls = os.path.join(basename, action_cls[i])
                input_list = []
                input_list += [os.path.join(basename_cls, f) for f in sorted(os.listdir(basename_cls)) if
                               f.split(".")[-1] == "mp4" or f.split(".")[-1] == "avi"]
                self.input_list.append(input_list)
                labels = [i%(len(action_cls)-1) for j in range(len(self.input_list[i]))] # handaction+armflapping same class
                self.labels.append(labels)

                gt_cls = os.path.join(basename, action_cls[i]) + '_per_frame_periodicity_colorful'
                self.gt_list += [os.path.join(gt_cls, f) for f in sorted(os.listdir(gt_cls)) if f.split(".")[-1] == "npy"]

            self.input_list = list(itertools.chain.from_iterable(self.input_list))
            self.labels = list(itertools.chain.from_iterable(self.labels))

        self.transform = transform
        self.root_dir = root_dir
        self.stage = stage
        self.mode = stage
        self.args = args
        self.video_transform_flag = True

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):

        rgbpath = self.input_list[idx]

        if len(self.gt_list) != 0:
            gtpath = self.gt_list[idx]
        else:
            gtpath = None
        label = self.labels[idx]

        video, bbox_video = load_video_v2(rgbpath, self.args)

        video_gt = np.load(gtpath)
        video_gt = video_gt.astype(np.float64)

        ## size matching
        video = video[:video_gt.size, :, :, :]
        video = video.astype(np.float32)

        #RepNet은 pre-process 있어서 끔
        if self.video_transform_flag == True:
            video = video_transform(video)

        sample = {'rgb': video, 'gt': video_gt}

        sample['label'] = label
        sample['rgbpath'] = rgbpath
        sample['bbox_video'] = bbox_video

        return sample



def load_video_v2(path, args):

    cap = cv2.VideoCapture(path)
    video_224 = []
    bbox_video = []

    fr_idx = 0

    while True:
        try:

            ret, frame = cap.read()
            if not ret:
                if fr_idx==0: # 첫프레임에 빈거 가끔 나옴
                    continue
                else:
                    break

            frame_224 = cv2.resize(frame, dsize=(224, 224)) #for repnet
            video_224.append(frame_224)

            fr_idx += 1

        except EOFError:
            break


    cap.release()

    # list to ndarray
    video_224 = np.array(video_224)

    return video_224, bbox_video


def smooth_detection(detections, window_size=5):
    """
    Smooths frame-level detections using a simple moving average.

    Args:
    - detections (list): List of frame-level detections (e.g., bounding box coordinates).
    - window_size (int): Size of the moving average window.

    Returns:
    - smoothed_detections (list): Smoothed frame-level detections.
    """
    smoothed_detections = []
    for i in range(len(detections)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(detections), i + window_size // 2 + 1)
        smoothed_value = np.mean(detections[start_idx:end_idx], axis=0)
        smoothed_detections.append(smoothed_value)
    return smoothed_detections


def NormalizeLen(rgb, vid_len=64):
    """ Return a normalized number of frames. """
    #rgb = rgb.cpu().detach().numpy()

    num_frames_rgb = rgb.shape[1]
    indices_rgb = np.linspace(0, num_frames_rgb-1, vid_len).astype(int)
    rgb = rgb[:, indices_rgb, :, :, :]

    # to tensor
    rgb = torch.from_numpy(rgb).cuda()

    return rgb


def video_transform(np_clip): #from imagent

    # Div by 255
    np_clip /= 255.

    # Normalization
    np_clip -= np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # mean
    np_clip /= np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 3)  # std

    return np_clip


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb, gt = sample['rgb'], sample['gt']
        return {'rgb': torch.from_numpy(rgb.astype(np.float32)),
                'gt': torch.from_numpy(np.asarray(gt))}


class VisualCenterCrop(object):
    """ Return a normalized number of frames. """

    def __init__(self, cropsize=(224, 224), central=True):
        self.cropsize = cropsize
        self.central = central

    def __call__(self, sample):
        rgb,  gt = sample['rgb'], sample['gt']
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
                'gt': gt}

