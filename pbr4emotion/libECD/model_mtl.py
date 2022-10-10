"""
                                      
   * Source: model_mtl.py
   * License: PBR License (Dual License)
   * Created by ByungOk Han <byungok.han@etri.re.kr> on 2020-07-30
   * Modified on 2022-09-30
   * Copyright 2022. ETRI all rights reserved. 
   
"""

# -*- coding: utf-8 -*- 



import torch
import torch.nn as nn

def conv1x1(in_planes, out_planes, stride=1):

    """conv1x1 function

    Note: conv1x1 function

    """
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MTL_Baseline(nn.Module):

    """MTL_Baseline class

    Note: MTL_Baseline class

    """

    def __init__(self, feature_model, feature_dim, num_emo_classes):

        """__init__ function

        Note: __init__ function

        """

        super(MTL_Baseline, self).__init__()
        self.feature_model = feature_model
        self.emo_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(), 
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Dropout(),
            nn.Linear(256, num_emo_classes)
            )
        self.val_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(), 
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Dropout(),
            nn.Linear(256, 1)
            )
        self.aro_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(), 
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Dropout(),
            nn.Linear(256, 1)
            )


    def forward(self, x):

        """forward function

        Note: __init__ function

        """

        img_feat = self.feature_model(x)
        emo_res  = self.emo_classifier(img_feat)
        val_res  = self.val_classifier(img_feat)
        aro_res  = self.aro_classifier(img_feat)


        return emo_res, val_res, aro_res
