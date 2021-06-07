# -*- coding: utf-8 -*- 

""" 
   * Source: libFER.model_cda.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalLayerF(Function):

    """GradientReversalLayerF class

    Note: class for GradientReversalLayerF

    """
    
    @staticmethod
    def forward(ctx, x, alpha):

        """forward function

        Note: function for forward

        """

        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):

        """backward function

        Note: function for backward

        """

        output = grad_output.neg() * ctx.alpha
        return output, None


class CrossDatasetAdaptation(nn.Module):

    """CrossDatasetAdaptation class

    Note:   class for CrossDatasetAdaptation
            a model for cross-dataset adaptation using a gradient reversal layer 

    """

    def __init__(self, feature_model, num_classes, num_ds_classes, with_label=False):

        """__init__ function

        Note: function for __init__

        """

        super(CrossDatasetAdaptation, self).__init__()
        self.feature_model = feature_model
        self.fc = nn.Linear(feature_model.get_feature_dim(), num_classes)

        self.with_label = with_label
        if self.with_label:
            self.fc_ds = nn.Linear(feature_model.get_feature_dim() + num_classes, num_ds_classes)
        else:
            self.fc_ds = nn.Linear(feature_model.get_feature_dim(), num_ds_classes)

    def forward(self, x, alpha=0.0, src_one_hot_label=None):

        """forward function

        Note: function for forward

        """

        return self.forward_impl(x, alpha, src_one_hot_label)

    def forward_impl_mtl(self, x):


        """forward_impl_mtl function

        Note: function for forward_impl_mtl

        """

        x = self.feature_model(x)
        x_ = x.clone()
        x = self.fc(x)
        x_ = self.fc_ds(x_)

        return x, x_

    def forward_impl(self, x, alpha, src_one_hot_label):

        """forward_impl function

        Note: function for forward_impl

        """

        x = self.feature_model(x)
        if self.with_label:
            x_ = torch.cat((x.clone(), src_one_hot_label), 1)
        else:
            x_ = x.clone()

        x = self.fc(x)
        x_ = GradientReversalLayerF.apply(x_, alpha)
        x_ = self.fc_ds(x_)

        return x, x_


class CrossDatasetAdaptation_Relu(nn.Module):

    """CrossDatasetAdaptation_Relu class

    Note:   class for CrossDatasetAdaptation_Relu
            cross-dataset adaptation model using Relu layers

    """

    def __init__(self, feature_model, num_classes, num_ds_classes, with_label=False):

        """__init__ function

        Note: function for __init__

        """

        super(CrossDatasetAdaptation_Relu, self).__init__()
        self.feature_model = feature_model
        self.fc = nn.Linear(feature_model.get_feature_dim(), 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)     
        self.with_label = with_label
        self.drop_fc = nn.Dropout2d(0.25)
        self.drop_fc1 = nn.Dropout2d(0.25)

        if self.with_label:
            self.fc_ds = nn.Linear(feature_model.get_feature_dim() + num_classes, 512)
        else:
            self.fc_ds = nn.Linear(feature_model.get_feature_dim(), 512)
         
        self.fc1_ds = nn.Linear(512, 256)    
        self.fc2_ds = nn.Linear(256, num_ds_classes)
        self.drop_fc_ds = nn.Dropout2d(0.25)
        self.drop_fc1_ds = nn.Dropout2d(0.25)


    def forward(self, x, alpha=0.0, src_one_hot_label=None):

        """forward function

        Note: function for forward

        """

        x = self.feature_model(x)
        if self.with_label:
            x_ = torch.cat((x.clone(), src_one_hot_label), 1)
        else:
            x_ = x.clone()

        # typical classifier
        x = F.leaky_relu(self.drop_fc(self.fc(x)))
        x = F.leaky_relu(self.drop_fc1(self.fc1(x)))
        x = self.fc2(x)

        # dataset classifier
        x_ = GradientReversalLayerF.apply(x_, alpha)
        x_ = F.leaky_relu(self.drop_fc_ds(self.fc_ds(x_)))
        x_ = F.leaky_relu(self.drop_fc1_ds(self.fc1_ds(x_)))
        x_ = self.fc2_ds(x_)

        return x, x_


def conv1x1(in_planes, out_planes, stride=1):

    """conv1x1 function

    Note: function for conv1x1

    """

    # 1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CrossDatasetAdaptation_Conv(nn.Module):

    """CrossDatasetAdaptation_Conv class

    Note:   class for CrossDatasetAdaptation_Conv
            cross-dataset adaptation model using Convolution layers

    """

    def __init__(self, feature_model, num_classes, num_ds_classes, with_label=False):

        """__init__ function

        Note: function for __init__

        """

        super(CrossDatasetAdaptation_Conv, self).__init__()
        self.feature_model = feature_model
        self.conv1 = conv1x1(feature_model.get_feature_dim(), 512)
        self.conv2 = conv1x1(512, 256)
        self.conv3 = conv1x1(256, num_classes)

        self.with_label = with_label
        if self.with_label:
            self.conv1_ds = conv1x1(feature_model.get_feature_dim() + num_classes, 512)
        else:
            self.conv1_ds = conv1x1(feature_model.get_feature_dim(), 512)
        self.conv2_ds = conv1x1(512, 256)
        self.conv3_ds = conv1x1(256, num_ds_classes) 


    def forward(self, x, alpha=0.0, src_one_hot_label=None):

        """forward function

        Note: function for forward

        """

        x = self.feature_model(x)
        x_ = x.clone()

        x = x.view(x.size(0), x.size(1), 1, 1)
        x_ = x_.view(x_.size(0), x_.size(1), 1, 1)

        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(x.size(0), x.size(1))

        x_ = GradientReversalLayerF.apply(x_, alpha)
        x_ = F.leaky_relu(self.conv1_ds(x_))
        x_ = F.leaky_relu(self.conv2_ds(x_))
        x_ = self.conv3_ds(x_)
        x_ = x_.view(x_.size(0), x_.size(1))

        return x, x_