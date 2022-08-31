""" 
   * Source: FGD_predict_gaze_test.py
   * License: PBR License (Dual License)
   * Modified by Howon Kim <hw_kim@etri.re.kr>
   * Date: 27 Jul 2022, ETRI
   * Copyright 2022. ETRI all rights reserved. 
"""

import cv2
import numpy as np
from FGD_api import PBR_FGD_predict_gaze 
from libFGD.utils import plot_kpts, plot_axis, get_projected_points, get_projected_axis, vector_to_pitchyaw, plot_gaze


def main():

    """main function

    Note: main function for facial 3D gaze with perspective projection model

    """
    
    cv_img = cv2.imread('./libFGD/test_image2.png')  #ref. from BIWI dataset
    cv2.imshow('input', cv_img)
    cam_mtx = np.array([[517.679, 0, 320], 
                        [0, 517.679, 240.5], 
                        [0, 0, 1]])

    
    # test PBR_FPD_predict_persp
    success, ret = PBR_FGD_predict_gaze(cv_img, cam_mtx)
    pred_lm2D = ret['lm2D']
    pred_lm3D = ret['lm3D']
    pred_rmtx = ret['rmtx'] 
    pred_tvec = ret['tvec']
    pred_gaze = ret['gaze']
    
    
    out_img = cv_img.copy() 
    out_img = plot_kpts(out_img, pred_lm2D, (0, 255, 0))
    
    axis_2D_pred = get_projected_axis(cam_mtx, pred_rmtx, pred_tvec)
    out_img = plot_axis(out_img, axis_2D_pred, 'pred') 
    
    lm2Ds_cam = get_projected_points(cam_mtx, pred_rmtx, pred_tvec, pred_lm3D)[:, 0:2]
    out_img = plot_kpts(out_img, lm2Ds_cam, (255, 0, 0))     
    
    pos2D_eyecenter = ((pred_lm2D[1]+pred_lm2D[2])*0.5).astype(np.int32)
    pred_gaze_py = vector_to_pitchyaw(pred_gaze*-1)
    out_img = plot_gaze(out_img, pos2D_eyecenter, pred_gaze_py, 3, (255, 255, 255))  
                        
                        
    cv2.imshow('FGD_gaze', out_img)
    cv2.waitKey(-1)


if __name__=='__main__':
    main()