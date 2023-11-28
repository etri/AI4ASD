"""
                                      
   * Source: offline_cpd_algorithms.py
   * License: PBR License (Dual License)
   * Created by ByungOk Han  <byungok.han@etri.re.kr> on 2022-02-18
   * Modified on 2023-11-20
   * Copyright 2023. ETRI all rights reserved. 
                                       
"""

import ruptures as rpt


def detect_change_points_pelt(signal, time_stamps, pen, 
                              model='rbf', min_size=2, jump=5):

    """
        detect_change_points_pelt function

        Note: detect_change_points_pelt function for ECD task

    """

    algo = rpt.Pelt(model=model, min_size = min_size, jump = jump).fit(signal)
    result = algo.predict(pen=pen)
    time_result = []
    for i in result:
        time_result.append(time_stamps[i-1])
    
    return time_result, result


def detect_change_points_binseg(signal, time_stamps,
                                pen=None, n_bkps=None, 
                                model='rbf', min_size=2, jump=5):

    """detect_change_points_binseg function

    Note: detect_change_points_binseg function for ECD task

    """

    algo = rpt.Binseg(model=model, min_size=min_size, jump=jump).fit(signal)
    result = algo.predict(n_bkps=n_bkps, pen=pen)
    time_result = []
    for i in result:
        time_result.append(time_stamps[i-1])

    return time_result, result


def detect_change_points_window(signal, time_stamps,
                                pen=None, n_bkps=None, 
                                win_width=20, model='rbf', min_size=2, jump=5):

    """detect_change_points_window function

    Note: detect_change_points_window function for ECD task

    """

    algo = rpt.Window(width=win_width, model=model, min_size=min_size, jump=jump).fit(signal)
    result = algo.predict(n_bkps=n_bkps, pen=pen)
    time_result = []
    for i in result:
        time_result.append(time_stamps[i-1])

    return time_result, result


def detect_change_points_dynp(signal, time_stamps, n_bkps,
                              model='rbf', min_size=2, jump=5):

    """detect_change_points_dynp function

    Note: detect_change_points_dynp function for ECD task

    """

    algo = rpt.Dynp(model=model, min_size = min_size, jump=jump).fit(signal)
    result = algo.predict(n_bkps=n_bkps)
    time_result = []
    for i in result:
        time_result.append(time_stamps[i-1])

    return time_result, result


def detect_change_points_bottomup(signal, time_stamps,
                                  pen=None, n_bkps=None, 
                                  model='rbf', min_size=2, jump=1):

    """detect_change_points_bottomup function

    Note: detect_change_points_bottomup function for ECD task

    """

    algo = rpt.BottomUp(model=model, min_size=min_size, jump=jump).fit(signal)
    result = algo.predict(n_bkps=n_bkps, pen=pen)
    time_result = []
    for i in result:
        time_result.append(time_stamps[i-1])

    return time_result, result


def detect_change_points_kernel(signal, time_stamps, 
                                pen=None, n_bkps=None, 
                                model='rbf', min_size=2, jump=1):

    """detect_change_points_kernel function

    Note: detect_change_points_kernel function for ECD task

    """

    algo = rpt.KernelCPD (model=model, min_size=min_size, jump=jump).fit(signal)
    result = algo.predict(n_bkps=n_bkps, pen=pen)
    time_result = []
    for i in result:
        time_result.append(time_stamps[i-1])

    return time_result, result



