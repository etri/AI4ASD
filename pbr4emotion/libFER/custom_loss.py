""" 
   * Source: libFER.custom_loss.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI
   * Copyright 2022. ETRI all rights reserved. 

"""

import torch
import torch.nn as nn

import numpy as np

_EPSILON =  1e-5


class AdaptiveCrossEntropyLoss(nn.Module):

    """Adaptive Cross Entropy Loss

    Note: Loss for unbalanced class samples

    """

    def __init__(self):
        
        """__init__ function

        Note: function for __init__

        """
        
        super(AdaptiveCrossEntropyLoss, self).__init__()

    def forward(self, logits, one_hot_labels, batch_cfm, var_factor):
        
        """forward function
           Args:
                logits: output of models
                one_hot_labels: labels of one hot vectors
                batch_cfm: confusion matrix per batch
                var_factor: variance factor of weight changes
        
            Returns:
                loss value       
        
        """
        
        class_counts = torch.sum(batch_cfm, axis=1) + _EPSILON
        batch_size = torch.sum(batch_cfm)
        num_classes = float(batch_cfm.shape[0])

        # distribution_compensation_term
        dist_comp_weight = (1.0 - class_counts / batch_size) / (num_classes - 1.0)
        dist_comp_weight = dist_comp_weight * num_classes + _EPSILON
        #print('dist_comp_weight=', dist_comp_weight)

        # precision_compensation_term
        prec_comp_weight = (1.0 - torch.diagonal(batch_cfm)/class_counts) 
        prec_comp_weight = prec_comp_weight / torch.sum(prec_comp_weight) * num_classes + _EPSILON
        #print('prec_comp_weight=', prec_comp_weight)

        adaptive_weight = dist_comp_weight * prec_comp_weight + _EPSILON

        mean_weight = torch.mean(adaptive_weight)
        diff_weight = adaptive_weight - mean_weight
        adaptive_weight = mean_weight + diff_weight * var_factor
        if var_factor == 0.0:
            adaptive_weight = 1.0

        log_prob = logits.log_softmax(dim=1)
        mean_cross_entropy = torch.mean(-1.0 * one_hot_labels * log_prob, axis=0)
        adaptive_cross_entropy = torch.sum(adaptive_weight * mean_cross_entropy)
        
        return adaptive_cross_entropy