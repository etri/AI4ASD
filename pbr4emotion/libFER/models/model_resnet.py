# -*- coding: utf-8 -*- 

""" 
   * Source: libFER.models.model_resnet.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI

"""

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):

    """conv3x3 function

    Note: function for conv3x3

    """

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):

    """conv1x1 function

    Note: function for conv1x1

    """

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    """BasicBlock class

    Note: class for BasicBlock

    """

    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):

        """__init__ function

        Note: function for __init__

        """

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        """forward function
        
        Note: function for forward
        
        """
        
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        
        out = self.relu(out)

        return out


class BasicBlock_v2(nn.Module):

    """BasicBlock_v2 class

    Note: class for BasicBlock_v2

    """

    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):

        """__init__ function

        Note: function for __init__

        """

        super(BasicBlock_v2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')

        self.bn1 = norm_layer(inplanes, momentum=0.997)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = norm_layer(planes, momentum=0.997)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        """forward function

        Note: function for forward

        """

        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        return out


class Bottleneck(nn.Module):

    """Bottleneck class

    Note: class for Bottleneck

    """

    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):

        """__init__ function

        Note: function for __init__

        """

        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):

        """forward function

        Note: function for forward

        """

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck_v2(nn.Module):

    """Bottleneck_v2 class

    Note: class for Bottleneck_v2

    """

    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):

        """__init__ function

        Note: function for __init__

        """

        super(Bottleneck_v2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        self.bn1 = norm_layer(inplanes, momentum=0.997)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(inplanes, width)
        self.bn2 = norm_layer(width, momentum=0.997)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn3 = norm_layer(width, momentum=0.997)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):

        """forward function

        Note: function for forward

        """

        identity = x
        
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class ResNet(nn.Module):

    """ResNet class

    Note: class for ResNet

    """


    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_width_dilation=None,
                 norm_layer=None, resnet_version=2, dim_add_vec = 0):

        """__init__ function

        Note: function for __init__

        """

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_width_dilation is None:
            replace_stride_width_dilation = [False, False, False]
        if len(replace_stride_width_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.resnet_version = resnet_version
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if self.resnet_version == 1:
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate = replace_stride_width_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_width_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_width_dilation[2])
        if self.resnet_version == 2:
            self.bn_pre = norm_layer(512 * block.expansion, momentum=0.997)
            self.relu_pre = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion + dim_add_vec, num_classes)
        self.org_feature_dim = 512 * block.expansion + dim_add_vec
        self.feature_dim = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.resnet_version == 1:
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)
        
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):

        """_make_layer function

        Note: function for _make_layer

        """

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate: 
            self.dilation *= stride
            stride=1
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, momentum=0.997),
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x, additional_vec = None):

        """_forward_impl function

        Note: function for _forward_impl

        """

        x = self.conv1(x)
        if self.resnet_version == 1:
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.resnet_version == 2:
            x = self.bn_pre(x)
            x = self.relu_pre(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        self.feature_vec = x
        if additional_vec != None:
            x = torch.cat((x, additional_vec), 1)
        x = self.fc(x)

        return x
       
    def forward(self, x, additional_vec=None):

        """forward function

        Note: function for forward

        """

        return self._forward_impl(x, additional_vec)

    def get_feature_dim(self):

        """get_feature_dim function

        Note: function for get_feature_dim

        """

        return self.feature_dim

    def get_org_feature_dim(self):

        """get_org_feature_dim function

        Note: function for get_org_feature_dim

        """

        return self.org_feature_dim


def _resnet(arch, block, layers, pretrained, progress, **kwargs):

    """_resnet function

    Note: function for _resnet

    """

    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):

    """resnet18 function

    Note: function for resnet18

    """

    kwargs['resnet_version'] = 2

    if kwargs['resnet_version'] == 1:
        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)
    elif kwargs['resnet_version'] == 2:
        return _resnet('resnet18', BasicBlock_v2, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):

    """resnet34 function

    Note: function for resnet34

    """

    kwargs['resnet_version'] = 2

    if kwargs['resnet_version'] == 1:
        return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)
    elif kwargs['resnet_version'] == 2:
        return _resnet('resnet34', BasicBlock_v2, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):

    """resnet50 function

    Note: function for resnet50

    """

    kwargs['resnet_version'] = 2

    if kwargs['resnet_version'] == 1:
        return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
    elif kwargs['resnet_version'] == 2:
        return _resnet('resnet50', Bottleneck_v2, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):

    """resnet101 function

    Note: function for resnet101

    """

    kwargs['resnet_version'] = 2

    if kwargs['resnet_version'] == 1:
        return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
    elif kwargs['resnet_version'] == 2:
        return _resnet('resnet101', Bottleneck_v2, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):

    """resnet152 function

    Note: function for resnet152

    """

    kwargs['resnet_version'] = 2

    if kwargs['resnet_version'] == 1:
        return _resnet('resnet152', Bottleneck, [3, 4, 36, 3], pretrained, progress, **kwargs)
    elif kwargs['resnet_version'] == 2:
        return _resnet('resnet152', Bottleneck_v2, [3, 4, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):

    """resnext50_32x4d function

    Note: function for resnext50_32x4d

    """

    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    kwargs['resnet_version'] = 2

    if kwargs['resnet_version'] == 1:
        return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
    elif kwargs['resnet_version'] == 2:
        return _resnet('resnext50_32x4d', Bottleneck_v2, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):

    """resnext101_32x8d function

    Note: function for resnext101_32x8d

    """

    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    kwargs['resnet_version'] = 2

    if kwargs['resnet_version'] == 1:
        return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
    elif kwargs['resnet_version'] == 2:
        return _resnet('resnext101_32x8d', Bottleneck_v2, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):

    """wide_resnet50_2 function

    Note: function for wide_resnet50_2

    """

    kwargs['width_per_group'] = 64 * 2
    kwargs['resnet_version'] = 2

    if kwargs['resnet_version'] == 1:
        return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
    elif kwargs['resnet_version'] == 2:
        return _resnet('wide_resnet50_2', Bottleneck_v2, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):

    """wide_resnet101_2 function

    Note: function for wide_resnet101_2

    """

    kwargs['width_per_group'] = 64 * 2
    kwargs['resnet_version'] = 2

    if kwargs['resnet_version'] == 1:
        return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
    elif kwargs['resnet_version'] == 2:
        return _resnet('wide_resnet101_2', Bottleneck_v2, [3, 4, 23, 3], pretrained, progress, **kwargs)