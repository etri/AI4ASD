""" 
   * Source: libFPD.api.py
   * License: PBR License (Dual License)
   * Modified by Howon Kim <hw_kim@etri.re.kr>
   * Date: 15 Nov 2021, ETRI

"""

import cv2
import numpy as np
from FPD_api import PBR_FPD_predict_persp 
from libFPD.utils import plot_kpts, plot_axis, get_projected_points, get_projected_axis


def main():

    """main function

    Note: main function for facial 3D pose with perspective projection model and orthographic projection model

    """
    
    cv_img = cv2.imread('./libFPD/test_image2.png')  #ref. from BIWI dataset
    cv2.imshow('input', cv_img)
    cam_mtx = np.array([[517.679, 0, 320], 
                        [0, 517.679, 240.5], 
                        [0, 0, 1]])

    
    # test PBR_FPD_predict_persp
    success, ret = PBR_FPD_predict_persp(cv_img, cam_mtx)
    pred_lm2D = ret['lm2D']
    pred_lm3D = ret['lm3D']
    pred_rmtx = ret['rmtx'] 
    pred_tvec = ret['tvec']
    
    
    out_img = cv_img.copy() 
    out_img = plot_kpts(out_img, pred_lm2D, (0, 255, 0))
    
    get_projected_axis(cam_mtx, pred_rmtx, pred_tvec)
    
    axis_2D_pred = get_projected_axis(cam_mtx, pred_rmtx, pred_tvec)
    out_img = plot_axis(out_img, axis_2D_pred, 'pred') 
    
    lm2Ds_cam = get_projected_points(cam_mtx, pred_rmtx, pred_tvec, pred_lm3D)[:, 0:2]
    out_img = plot_kpts(out_img, lm2Ds_cam, (255, 0, 0))     
    cv2.imshow('FPD_persp', out_img)
    cv2.waitKey(-1)


if __name__=='__main__':
    main()