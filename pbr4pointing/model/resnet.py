"""
   * Source: resnet.py
   * License: PBR4AI License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 21 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""

import torch
from torch import nn
from torchvision import models

class Resnet50_Siam(nn.Module):

    """ Resnet50_Siam class

    Note: class for Resnet50_Siam

    """

    def __init__(self, num_classes=2):

        """ __init__ function for Resnet50_Siam class

        """

        super(Resnet50_Siam, self).__init__()
        net = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(net.children())[:-1])
        self.classifier = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1))
        self.h = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1)
        )

    def forward(self, x1, x2):

        """ forward function for Resnet50_Siam class

        """

        z1 = self.features(x1)
        z2 = self.features(x2)

        p1 = self.h(z1)
        p2 = self.h(z2)

        logits1 = self.classifier(z1)
        logits1 = logits1.squeeze(2)
        logits1 = logits1.squeeze(2)

        metric_feature = [z1, z2, p1, p2]

        return logits1, metric_feature
