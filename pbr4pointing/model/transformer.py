"""
   * Source: transformer.py
   * License: PBR4AI License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 21 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""

import timm
from torch import nn

class vit_Siam(nn.Module):

    """ Resnet50_Siam class

    Note: class for Resnet50_Siam

    """

    def __init__(self, num_classes=2, fea_dim=768):

        """ __init__ function for vit_Siam class

        """

        super(vit_Siam, self).__init__()
        self.backbone = timm.create_model(
            'vit_base_patch32_224',
            pretrained=True,
            num_classes=2
        )

        self.in_features = fea_dim
        self.latent_feature = int(fea_dim / 4)

        self.backbone.head = nn.Identity()

        self.h = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.latent_feature),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.latent_feature, out_features=self.in_features)
        )

        self.classifier = nn.Linear(in_features=self.in_features, out_features=2)

    def forward(self, x1, x2):

        """ forward function for vit_Siam class

        """

        z1 = self.backbone(x1)
        z2 = self.backbone(x2)

        p1 = self.h(z1)
        p2 = self.h(z2)

        logits1 = self.classifier(z1)
        metric_feature = [z1, z2, p1, p2]

        return logits1, metric_feature

