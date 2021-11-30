""" 
   * Source: libFPD.api.py
   * License: PBR License (Dual License)
   * Modified by Howon Kim <hw_kim@etri.re.kr>
   * Date: 15 Nov 2021, ETRI

"""

import cv2
from FPD_api import PBR_FPD_predict_ortho
from libFPD.utils import plot_kpts, plot_axis


def main():

    """main function

    Note: main function for facial 3D pose with perspective projection model and orthographic projection model

    """
    
    cv_img = cv2.imread('./libFPD/test_image2.png')  #ref. from BIWI dataset
    cv2.imshow('input', cv_img)
    
    # test PBR_FPD_predict_ortho
    success, ret = PBR_FPD_predict_ortho(cv_img)
    pred_rmtx_ortho = ret['rmtx_ortho']
    pred_lm2D       = ret['lm2D']
    pred_axis       = ret['axis']
        
    out_img = cv_img.copy() 
    out_img = plot_axis(out_img, pred_axis)
    out_img = plot_kpts(out_img, pred_lm2D)
    cv2.imshow('FPD_ortho', out_img)
    cv2.waitKey(-1)


if __name__=='__main__':
    main()