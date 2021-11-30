""" 
   * Source: libFPD.api.py
   * License: PBR License (Dual License)
   * Modified by Howon Kim <hw_kim@etri.re.kr>
   * Date: 15 Nov 2021, ETRI

"""

from libFD.api import PBR_libFD_predict
from libFPD.api import PBR_libFPD_predict_ortho, PBR_libFPD_predict_persp


def PBR_FPD_predict_ortho(cv_img):

    """ facial 3D pose detection main api with the orthographic projection model

    Args:
        cv_img: opencv loaded image hxwx3       
    """   
    
    face_regions = PBR_libFD_predict(cv_img)    
    if face_regions.shape[0]==0:
        return False, None
        
    outputs = PBR_libFPD_predict_ortho(cv_img, face_regions[0][0:4])    
        
    return True, outputs


def PBR_FPD_predict_persp(cv_img, cam_mtx):

    """ facial 3D pose detection main api with the perspective projection model (require the camera intrinsic matrix)

    Args:
        cv_img: opencv loaded image hxwx3     
        cam_mtx: 3x3 input image's intrinsic matrix 
    """   
    
    face_regions = PBR_libFD_predict(cv_img)
    
    if face_regions.shape[0]==0:
        return False, None

    outputs = PBR_libFPD_predict_persp(cv_img, face_regions[0][0:4], cam_mtx)    
        
    return True, outputs


