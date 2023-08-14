"""
   * Source: RepDetectModel.py
   * License: PBR License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 3 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""


from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():

    """get_inplanes function

    """

    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):

    """conv3x3x3 function

    Note: function for conv3x3x3

    """

    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):

    """conv1x1x1 function

    Note: function for conv1x1x1

    """

    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):

    """BasicBlock class

    Note: class for BasicBlock

    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):

        """__init__ function for BasicBlock class

        """

        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        """forward function for BasicBlock class

        """

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    """Bottleneck class

    Note: class for Bottleneck

    """

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):

        """__init__ function for Bottleneck class

        """

        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        """forward function for Bottleneck class

        """

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    """ResNet class

    Note: class for ResNet

    """

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1039):

        """__init__ function for ResNet class

        """

        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):

        """_downsample_basic_block function for ResNet class

        """

        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):

        """_make_layer function for ResNet class

        """

        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        """forward function for ResNet class

        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):

    """generate_model function

    Note: function for generating  ResNet model

    """

    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


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

    """Up_conv class

    Note: Upscaling then double conv

    """

    def __init__(self, in_channels, out_channels):

        """__init__ function for up_conv class

        """

        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d_conv = double_conv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):

        """forward function for up_conv class

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

    """RepDetectNet_3D_base_s2 class

    Note: main architecture for detecting temporal segments in the video

    Arguments:
        rgb (opencv image) : video

    Returns:
        logits : predicted output binary matrix

    """

    def __init__(self):

        """__init__ function for RepDetectNet_3D_base_s2 class

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

        """forward function for RepDetectNet_3D_base_s2 class

        """

        x = x.permute(0, 4, 1, 2, 3)

        x = self.base_net.conv1(x)
        x = self.base_net.bn1(x)
        x = self.base_net.relu(x)
        latent_feature = self.base_net.layer1(x)
        latent_feature2 = self.base_net.layer2(latent_feature)

        latent_feature = latent_feature.permute(0, 2, 1, 3, 4)
        latent_feature = torch.squeeze(latent_feature, 0)

        latent_feature2 = latent_feature2.permute(0, 2, 1, 3, 4)
        latent_feature2 = torch.squeeze(latent_feature2, 0)

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

        ##########
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

        return logits
