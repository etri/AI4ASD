"""
   * Source: smoothing_filters.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han  <byungok.han@etri.re.kr> on 2023-11-20
   * Copyright 2023. ETRI all rights reserved. 
                                       
"""

# -*- coding: utf-8 -*- 


from scipy.signal import savgol_filter
import cv2
import numpy as np


def savgol(signal, window_length, polyorder):

    """savgol function

    Note: savgol function for ECD task

    """

    x = signal.flatten()
    y = savgol_filter(x, window_length, polyorder)
    y = y.reshape((-1, 1))

    return y


def bilateral(signal, diameter, sigma_color):

    """bilateral function

    Note: bilateral function for ECD task

    """

    x = signal.flatten()
    y = cv2.bilateralFilter(np.expand_dims(np.expand_dims(x, 1), 
                                           2).astype(np.float32), 
                            -1, diameter, sigma_color)
    y = y.reshape((-1, 1))

    return y


def gaussian(signal, kernel_size, sigma):
    
    """gaussian function

    Note: gaussian function for ECD task

    """

    x = signal.flatten()
    g_kernel1d = cv2.getGaussianKernel(kernel_size, sigma)
    g_kernel2d = np.outer(g_kernel1d, g_kernel1d.transpose())
    y = cv2.filter2D(np.expand_dims(np.expand_dims(x, 1), 
                                           2).astype(np.float32), 
                            -1, g_kernel2d)
    y = y.reshape((-1, 1))

    return y