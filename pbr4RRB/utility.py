import time
import numpy as np
import cv2
import torch
import numpy as np
import numpy.ma as ma
import math
import operator
import torch.nn as nn
import copy
#import clip
import mxnet as mx
import cmapy
import itertools
import csv
import random

from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple
from pathlib import Path


IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])

tf_Resize = transforms.Resize((224, 224))
tf_ToTensor = transforms.ToTensor()
tf_Normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

action_cls = ['ArmFlapping', 'Spinning', 'HeadBanging']
action_color_list = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (128, 128, 128)] # Red, Yellow, Green, Blue, Purple, Gray


class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def clip_list(A, len_max):
    if len(A) > len_max:
        A = A[1:]

    return A


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def optimal_font_dims(img, font_scale = 2e-3, thickness_scale = 2e-3):
    h, w, _ = img.shape
    font_scale = min(w, h) * font_scale
    thickness = math.ceil(min(w, h) * thickness_scale)
    return font_scale, thickness


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def visualize_result(path, gt, args, result_list):

    """visualize_result function

    Note: function for visualizing inference result(temporal localization + action classification)

    """

    with torch.no_grad():

        cap = cv2.VideoCapture(path)
        num_frames = gt.shape[0]

        # Define the codec and create VideoWriter object
        global video_num_cnt

        for fr_idx in range(num_frames):

            flag, image_bgr = cap.read()
            if not flag:
                break

            if fr_idx == 0:

                color_bar_gt = np.full((20, num_frames, 3), (0, 0, 0), dtype=np.uint8)
                color_bar_pred = np.full((20, num_frames, 3), (0, 0, 0), dtype=np.uint8)

                for i in range(len(action_cls)):
                    color_bar_gt[:, np.where(gt == i + 1), 0] = action_color_list[i][0]
                    color_bar_gt[:, np.where(gt == i + 1), 1] = action_color_list[i][1]
                    color_bar_gt[:, np.where(gt == i + 1), 2] = action_color_list[i][2]

                if len(result_list) != 0:
                    for frame_idx in range(num_frames):
                        for i in range(len(result_list)):
                            if frame_idx >= result_list[i]['start_idx'] and frame_idx <= result_list[i]['end_idx']:

                                if result_list[i]['proposal_conf'] > args.proposal_conf_th and result_list[i]['prob'] > args.prob_conf_th:  ## add proposal_conf_threshold
                                    action_preds = result_list[i]['action_preds'][0]
                                    color_bar_pred[:, frame_idx, :] = action_color_list[action_preds][:]

                color_bar_paper = np.full((20, num_frames, 3), (127, 127, 127), dtype=np.uint8)  # (255, 255, 255)
                color_bar_paper2 = np.full((20, num_frames, 3), (127, 127, 127), dtype=np.uint8)  # (255, 255, 255)

            color_bar_gt_copy = color_bar_gt.copy()
            color_bar_pred_copy = color_bar_pred.copy()

            if fr_idx > 2 and fr_idx<num_frames-3:
                for i in [-3, -2, -1, 0, 1, 2, 3]:
                    color_bar_gt_copy[:, fr_idx + i, 0] = 255
                    color_bar_gt_copy[:, fr_idx + i, 1] = 0
                    color_bar_gt_copy[:, fr_idx + i, 2] = 0
                    color_bar_pred_copy[:, fr_idx + i, 0] = 255
                    color_bar_pred_copy[:, fr_idx + i, 1] = 0
                    color_bar_pred_copy[:, fr_idx + i, 2] = 0

            color_bar_cat = cv2.vconcat([color_bar_paper, color_bar_pred_copy])
            color_bar_cat = cv2.vconcat([color_bar_cat, color_bar_paper2])
            color_bar_cat = cv2.vconcat([color_bar_cat, color_bar_gt_copy])
            color_bar_cat = cv2.resize(color_bar_cat,
                                       dsize=(250 * args.viz_scale * 3,
                                       int(250 * args.viz_scale * 3 * 20 / num_frames*4)),
                                       interpolation=cv2.INTER_NEAREST)

            frame_resize = cv2.resize(image_bgr,
                                      dsize=(color_bar_cat.shape[1],
                                      int(image_bgr.shape[0] / image_bgr.shape[1] * color_bar_cat.shape[1])),
                                      interpolation=cv2.INTER_LINEAR)

            y_anchor, x_anchor, _ = frame_resize.shape

            if len(result_list) != 0:
                for i in range(len(result_list)-1):
                    if fr_idx >= result_list[i]['start_idx'] and fr_idx <= result_list[i]['end_idx']:

                        action_preds = action_cls[result_list[i]['action_preds'][0]]

                        font_scale, thickness = optimal_font_dims(frame_resize)
                        text = action_preds + "(" + str(int(result_list[i]['prob'] * 1000) / 10) + '%)'
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,thickness)
                        text_w, text_h = text_size
                        pos = (int(x_anchor * 0.05), int(y_anchor * 0.7))
                        cv2.rectangle(frame_resize, pos, (pos[0] + text_w, pos[1] + text_h+10), (0, 255, 0), -1)
                        cv2.putText(frame_resize, text, (pos[0], pos[1] + text_h + 2 - 1),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

            result = cv2.vconcat([frame_resize, color_bar_cat])

            cv2.imshow('result', result)
            k = cv2.waitKey(1)
            if k == 27:  # esc key
                break

            torch.cuda.empty_cache()

        cap.release()

