""" 
   * Source: FGD_api.py
   * License: PBR License (Dual License)
   * Modified by Howon Kim <hw_kim@etri.re.kr>
   * Date: 27 Jul 2022, ETRI
   * Copyright 2022. ETRI all rights reserved. 
"""

from libFD.api import PBR_libFD_predict
from libFGD.api import PBR_libFGD_predict_gaze


def PBR_FGD_predict_gaze(cv_img, cam_mtx):

    """ facial 3D gaze detection main api with the perspective projection model (require the camera intrinsic matrix)

    Args:
        cv_img: opencv loaded image hxwx3     
        cam_mtx: 3x3 input image's intrinsic matrix 
    """   
    
    face_regions = PBR_libFD_predict(cv_img)
    
    if face_regions.shape[0]==0:
        return False, None

    outputs = PBR_libFGD_predict_gaze(cv_img, face_regions[0][0:4], cam_mtx)    
        
    return True, outputs


