"""
   * Source: utility.py
   * License: PBR4AI License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 21 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""

import time
import numpy as np
import cv2
import torch
import numpy as np
import numpy.ma as ma
import math
import operator
import torch.nn as nn
from typing import Optional, Tuple

class Adder(object):

    """Adder class

    Note: class for Adder

    """

    def __init__(self):

        """ __init__ function for Adder class

        """

        self.count = 0
        self.num = float(0)

    def reset(self):

        """reset function for Adder class

        """

        self.count = 0
        self.num = float(0)

    def __call__(self, num):

        """__call__ function for Adder class

        """

        self.count += 1
        self.num += num

    def average(self):

        """average function for Adder class

        """

        return self.num / self.count


def check_lr(optimizer):

    """ check_lr function

    Note: function for user check_lr

    """

    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr


def softmax(x):

    """ softmax function

    Note: function for softmax

    """

    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def transform_2d_to_3d(depth, x, y, camera_param):

    """ transform_2d_to_3d function

    Note: function for transform_2d_to_3d

    """

    image_h, image_w = depth.shape[:2]

    depth_roi = depth[max(0,y-20):min(image_h-1, y+20), max(0,x-20):min(image_w-1, x+20)]

    try:
        min_depth = np.min(ma.masked_where(depth_roi == 0, depth_roi))
        if min_depth >= 1:
            pass
        else:
            min_depth = 100000 #0

    except ValueError:
        min_depth = 0

    fx = camera_param['fx']
    fy = camera_param['fy']
    cx = camera_param['cx']
    cy = camera_param['cy']

    pt_3d = (min_depth / fx * (x - cx),
             min_depth / fy * (y - cy),
             float(min_depth))

    return pt_3d


def cal_dist(pt1, pt2):

    """ cal_dist function

    Note: function for cal_dist

    """

    dist_3d = math.sqrt(
        (pt1[0]-pt2[0]) * (pt1[0]-pt2[0]) + (pt1[1]-pt2[1]) * (pt1[1]-pt2[1]) + (pt1[2]-pt2[2]) * (pt1[2]-pt2[2]))

    return dist_3d


def loss_negative_cosine(p, z, label):

    """ loss_negative_cosine function

    Note: function for calculating SimSiam loss

    """

    # stop gradient
    z = z.detach()

    batch_size = p.shape[0]

    p_view = p.view(batch_size, -1)
    z_view = z.reshape(batch_size, -1)

    p_norm = torch.norm(p_view, dim=1)
    p_norm = torch.unsqueeze(p_norm, 1)
    z_norm = torch.norm(z_view, dim=1)
    z_norm = torch.unsqueeze(z_norm, 1)

    p_norm = p_view / p_norm
    z_norm = z_view / z_norm

    loss = -torch.mean(torch.sum(p_norm * z_norm, dim=1))

    return loss

