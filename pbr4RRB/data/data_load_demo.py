"""
   * Source: data_load_demo.py
   * License: PBR License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 3 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""

import os
import itertools
from utility import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def get_dataloaders(args):

    """get_dataloaders function

    Note: function for getting dataloaders

    """

    transformer_val = transforms.Compose([ToTensor()])
    dataset_test = DATA(args.datadir, transform=transformer_val, stage='test', args=args)
    datasets = {'test': dataset_test}

    dataloaders = {'test': DataLoader(datasets['test'], batch_size=1, shuffle=False,
                    num_workers=args.num_workers, drop_last=False, pin_memory=True)}
    return dataloaders


class DATA(Dataset):

    """DATA class

    Note: class for loading and pre-processing input data

    """

    def __init__(self, root_dir='',
                 transform=None,
                 stage='train',
                 args=None):

        """ __init__ function for DATA class

        """

        if args.data_choice == 'SSBD':

            action_cls = ['ArmFlapping', 'Spinning', 'HeadBanging']
            basename = os.path.join(root_dir, 'Dataset')

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

                gt_cls = os.path.join(basename, action_cls[i]) + '_per_frame_periodicity'
                self.gt_list += [os.path.join(gt_cls, f) for f in sorted(os.listdir(gt_cls)) if
                                 f.split(".")[-1] == "npy"]

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
                labels = [i%3 for j in range(len(self.input_list[i]))] # handactio & armflapping merge
                self.labels.append(labels)

                gt_cls = os.path.join(basename, action_cls[i]) + '_per_frame_periodicity'
                self.gt_list += [os.path.join(gt_cls, f) for f in sorted(os.listdir(gt_cls)) if
                                 f.split(".")[-1] == "npy"]

            self.input_list = list(itertools.chain.from_iterable(self.input_list))
            self.labels = list(itertools.chain.from_iterable(self.labels))

        self.transform = transform
        self.root_dir = root_dir
        self.stage = stage
        self.mode = stage
        self.args = args
        self.video_transform_flag = True

    def __len__(self):

        """ __len__ function for DATA class

        """

        return len(self.input_list)

    def __getitem__(self, idx):

        """ __getitem__ function for DATA class

        """

        rgbpath = self.input_list[idx]
        video = load_video(rgbpath)
        label = self.labels[idx]

        if len(self.gt_list) != 0:
            gtpath = self.gt_list[idx]
            video_gt = np.load(gtpath)
            video_gt = video_gt.astype(np.float64)
        else:
            video_gt = np.zeros_like(video)

        ## size matching
        video = video[:video_gt.size, :, :, :]

        if self.video_transform_flag == True:
            video = video_transform(video)

        sample = {'rgb': video, 'gt': video_gt}
        if self.transform:
            sample = self.transform(sample)

            # append rgb_path to sample dictionary
            sample['label'] = label
            sample['rgbpath'] = rgbpath

        return sample


def load_video(path):

    """load_video function

    Note: function for loading and re-sizing video data

    """

    cap = cv2.VideoCapture(path)
    video_224 = []
    fr_idx = 0

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                if fr_idx==0:
                    continue
                else:
                    break

            frame_224 = cv2.resize(frame, dsize=(224, 224))
            video_224.append(frame_224.astype(np.float32))
            fr_idx += 1

        except EOFError:
            break

    cap.release()
    video_224 = np.array(video_224)

    return video_224


def NormalizeLen(rgb):

    """NormalizeLen function

    Note: function for normalizing number of frames

    """

    vid_len = 64

    rgb = rgb.cpu().detach().numpy()

    num_frames_rgb = rgb.shape[1]
    indices_rgb = np.linspace(0, num_frames_rgb-1, vid_len).astype(int)
    rgb = rgb[:, indices_rgb, :, :, :]

    # to tensor
    rgb = torch.from_numpy(rgb).cuda()

    return rgb


def video_transform(np_clip):

    """video_transform function

    Note: function for normalizing pixel values of frames

    """

    # Div by 255
    np_clip /= 255.

    # Normalization
    np_clip -= np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # mean
    np_clip /= np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 3)  # std

    return np_clip


class ToTensor(object):

    """ToTensor function

    Note: function for converting ndarrays in sample to Tensors

    """

    def __call__(self, sample):
        rgb, gt = sample['rgb'], sample['gt']
        return {'rgb': torch.from_numpy(rgb.astype(np.float32)),
                'gt': torch.from_numpy(np.asarray(gt))}

