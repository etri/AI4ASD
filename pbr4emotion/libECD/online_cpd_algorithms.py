"""
   * Source: online_cpd_algorithms.py
   * License: PBR License (Dual License)
   * Created by ByungOk Han  <byungok.han@etri.re.kr> on 2022-02-18
   * Modified on 2023-11-20
   * Copyright 2023. ETRI all rights reserved. 
                                       
"""

import numpy as np
import bocd


def detect_change_points_bayesian(signal, time_stamps, run_length=20, thresh=0.0):

    """detect_change_points_bayesian function

    Note: detect_change_points_bayesian function for ECD task

    """

    bcs = []
    for j in range(signal.shape[1]):
        bc = bocd.BayesianOnlineChangePointDetection(
                bocd.ConstantHazard(run_length), 
                bocd.StudentT(mu=0, kappa=1, alpha=1, beta=1))
        bcs.append(bc)

    rt_mle = np.empty(signal.shape)
    #print(signal.shape)

    for j in range(signal.shape[1]):
        for i in range(signal.shape[0]):
            bcs[j].update(signal[i, j])
            rt_mle[i, j] = bcs[j].rt    

    rt_mle = np.average(rt_mle, axis = 1)
    print(rt_mle.shape)
    result = np.where(np.diff(rt_mle) < thresh)[0]
    print(np.diff(rt_mle))

    time_result = []
    for i in result:
        time_result.append(time_stamps[i-1])

    return time_result, result


# simple gradient based change point detection
# pen은 threshold 값으로 gradient의 절대값이 pen 값 보다 크면 
# change point로 detect 한다. 
def detect_change_points_gradient(signal, time_stamps, 
                                  pen=None, n_bkps=None):

    """detect_change_points_gradient function

    Note: detect_change_points_gradient function for ECD task

    """
    
    # 1 dimension이 줄어듬
    gradient_signal = np.diff(signal, axis=0)
    
    if pen!=None:
        result = np.where(np.abs(gradient_signal) > pen)[0]
    if n_bkps!=None:
        sorted_grad_idx = np.argsort(np.abs(gradient_signal), axis=None)
        #print(signal.shape)
        #print(gradient_signal.shape)
        #print(sorted_grad_idx.shape)
        result = np.squeeze(sorted_grad_idx)[:n_bkps]%gradient_signal.shape[0]

    gradient_time_stamps = []
    n = 0
    prev = 0
    for t in time_stamps:
        if n!=0:
            gradient_time_stamps.append((t + prev)/ 2.0)
        prev = t
        n = n+1

    time_result = []
    for i in result:
        time_result.append(gradient_time_stamps[i])

    return time_result, result
