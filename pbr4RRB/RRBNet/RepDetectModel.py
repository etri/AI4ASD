"""
   * Source: RepDetectModel.py
   * License: PBR License (Dual License)
   * Modified by CheolHwan Yoo <ch.yoo@etri.re.kr>
   * Date: 22 Jul. 2022, ETRI
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Callable
from torchvision.models import resnet50, wide_resnet50_2, resnet18, vgg19
import matplotlib.pyplot as pl
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ResNet3D.ResNet3D import *

import timm.models


def double_conv(in_channels, out_channels, mid_channels=None):

    """DoubleConv function
    Note: (convolution => [BN] => ReLU) * 2"
    """

    if not mid_channels:
        mid_channels = out_channels

    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class up_conv(nn.Module):
    """Up class
    Note: Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels):

        """__init__ function for Up class
        """

        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d_conv = double_conv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):

        """forward function for Up class
        """

        x1 = self.upsample(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.d_conv(x)


class RepDetectNet_3D_base_s2(nn.Module):

    """RepDetectNet_3D_multi_level_v2 class
    Note: main architecture for detection periodic segments in the video
    """

    def __init__(self, num_classes=60, dropout_drop_prob=0.5, input_channel=3, spatial_squeeze=True):

        """__init__ function for RepDetectNet_3D_multi_level_v2 class
        """

        super(RepDetectNet_3D_base_s2, self).__init__()

        self.base_net = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes())

        # UNet
        factor = 2
        self.inc = double_conv(1, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(512, 1024 // factor)
        )

        self.up1 = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(128, 256)
        )

        self.up1 = up_conv(1024, 512 // factor)
        self.up2 = up_conv(512, 256 // factor)
        self.up3 = up_conv(256, 128 // factor)
        self.up4 = up_conv(128, 64)
        self.outc = nn.Conv2d(64, 2, kernel_size=1)


    def forward(self, x):

        """forward function for RepDetectNet_3D_multi_level_v2 class
        """

        x = x.permute(0, 4, 1, 2, 3)

        #with torch.no_grad(): #### freeze encoder weights
        ############ ResNet18 3D 의 앞쪽 일부
        x = self.base_net.conv1(x)
        x = self.base_net.bn1(x)
        x = self.base_net.relu(x)
        latent_feature = self.base_net.layer1(x)
        latent_feature2 = self.base_net.layer2(latent_feature)

        latent_feature = latent_feature.permute(0, 2, 1, 3, 4)
        latent_feature = torch.squeeze(latent_feature, 0)

        latent_feature2 = latent_feature2.permute(0, 2, 1, 3, 4)
        latent_feature2 = torch.squeeze(latent_feature2, 0)


        #################
        seq_len = latent_feature.shape[0]
        embs = torch.reshape(latent_feature, [seq_len, -1])
        embs_transpose = torch.transpose(embs, 0, 1)
        ssm = torch.matmul(embs, embs_transpose)
        ssm = torch.unsqueeze(ssm, 0)
        ssm = torch.unsqueeze(ssm, 0)


        seq_len = latent_feature2.shape[0]
        embs2 = torch.reshape(latent_feature2, [seq_len, -1])
        embs_transpose2 = torch.transpose(embs2, 0, 1)
        ssm2 = torch.matmul(embs2, embs_transpose2)
        ssm2 = torch.unsqueeze(ssm2, 0)
        ssm2 = torch.unsqueeze(ssm2, 0)
        embs_norm = torch.sqrt(torch.sum(torch.square(embs2), dim=1))
        embs_norm = torch.reshape(embs_norm, [-1, 1])
        embs2 = embs2 / embs_norm
        embs_transpose2 = torch.transpose(embs2, 0, 1)
        temp2 = torch.matmul(embs2, embs_transpose2)
        temp2 = torch.unsqueeze(temp2, 0)
        temp2 = torch.unsqueeze(temp2, 0)

        ##########
        temp2 = F.interpolate(temp2, size=(ssm.shape[2], ssm.shape[3]), mode='bicubic', align_corners=False)
        ssm2 = F.interpolate(ssm2, size=(ssm.shape[2], ssm.shape[3]), mode='bicubic', align_corners=False)

        x1 = self.inc(ssm2)
        x1_latent = self.down1[0](x1)

        x2 = self.down1[1](x1_latent)
        x2_latent = self.down2[0](x2)

        x3 = self.down2[1](x2_latent)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits, temp2

