""" 
   * Source: libFGD.api.py
   * License: PBR License (Dual License)
   * Modified by Howon Kim <hw_kim@etri.re.kr>
   * Date: 27 Jul 2022, ETRI
   * Copyright 2022. ETRI all rights reserved. 
"""


from libFGD.facegaze import Face3DGaze




def PBR_libFGD_predict_gaze(cv_img, roi_box, cam_mtx):

    """PBR_libFPD_predict function for facial 3D pose estimation

    Note: libFPD Wrapper API

    Arguments: 
        cv_img (opencv image) : image to detect facial 3D pose
        roi_box : face bounding box [sx, sy, ex, ey]
        cam_mtx : 3x3 input image's intrinsic matrix  
    Returns:
        predicted_results(list) : {'lm2D', 'rmtx_ortho'}

    """

    model_path_facepose = './libFGD/weights/orthof3d_resnet50.pth'
    model_path_facegaze = './libFGD/weights/normf3d_resnet50.pth'
    fgaze=Face3DGaze(model_path_facepose, model_path_facegaze)    

    return fgaze.predict_gaze(cv_img, roi_box, cam_mtx)
