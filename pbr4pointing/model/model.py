"""
   * Source: model.py
   * License: PBR4AI License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 21 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""

from model.resnet import Resnet50_Siam
from model.transformer import vit_Siam
from BYOL.byol_pytorch import BYOL
from torchvision import models
import timm.models
import torch

def build_net(args):

    """build_net function

    Note: function for build_net

    """

    if args.backbone == 'resnet':
        if args.SSL == 'None':
            model = models.resnet50(pretrained=True)
            model.fc = torch.nn.Linear(2048, 2)

        elif args.SSL == 'SimSiam':
            model = Resnet50_Siam(2)

        elif args.SSL == 'BYOL':
            resnet = models.resnet50(pretrained=True)
            ## BYOL
            model = BYOL(
                resnet,
                image_size=256,
                pre_class_dim=2048,
                hidden_layer='avgpool',
                use_momentum=True  # turn off momentum in the target encoder
            )

    elif args.backbone == 'vit_B_32':
        fea_dim = 768

        if args.SSL == 'None':
            # Transformer ##
            model = timm.create_model(
                'vit_base_patch32_224',
                pretrained=True,
                num_classes=2
                )
        elif args.SSL == 'SimSiam':
            model = vit_Siam(2, fea_dim)
        elif args.SSL == 'BYOL':
            vit = timm.create_model(
                'vit_base_patch32_224',
                pretrained=True,
                num_classes=2
            )
            model = BYOL(
                vit,
                image_size=256,
                pre_class_dim=fea_dim,
                hidden_layer='pre_logits',
                use_momentum=True  # turn off momentum in the target encoder
            )

    return model
