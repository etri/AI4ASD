"""
   * Source: RepClsModel.py
   * License: PBR License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 3 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""

import torch.nn as nn


class VideoSwinTransformerModel(nn.Module):

    """ VideoSwinTransformerModel class

    Note: main architecture for recognizing(classifying) action class from proposed temporal segments

    Arguments:
        rgb (opencv image) : temporal segments

    Returns:
        logits : predicted logit values

    """

    def __init__(self, mode=None):

        """__init__ function for VideoSwinTransformerModel class

        """

        super(VideoSwinTransformerModel, self).__init__()

        #from mmcv import Config
        from mmengine.config import Config
        from mmaction.models import build_model
        self.mode = mode

        config = 'configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py'
        cfg = Config.fromfile(config)
        self.model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        self.head = nn.Linear(1024, 3)

    def forward(self, rgb):

        """forward function for VideoSwinTransformerModel class

        """

        rgb = rgb.permute(0, 4, 1, 2, 3)
        feat = self.model.backbone(rgb)

        # mean pooling
        feat = feat.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]

        # final output
        logits = self.head(feat)

        return logits