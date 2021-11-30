""" 
   * Source: libFPD.api.py
   * License: PBR License (Dual License)
   * Modified by Howon Kim <hw_kim@etri.re.kr>
   * Date: 15 Nov 2021, ETRI

"""


from libFPD.facepose import Face3DPose


def PBR_libFPD_predict_ortho(cv_img, roi_box):

    """PBR_libFPD_predict function for facial 3D pose estimation

    Note: libFPD Wrapper API

    Arguments: 
        cv_img (opencv image) : image to detect facial 3D pose
        roi_box : face bounding box [sx, sy, ex, ey]
    Returns:
        predicted_results(list) : {'lm2D', 'rmtx_ortho'}

    """
    model_path='./libFPD/weights/orthof3d_resnet50.pth'
    fpose=Face3DPose(model_path)    


    return fpose.predict_ortho(cv_img, roi_box)


def PBR_libFPD_predict_persp(cv_img, roi_box, cam_mtx):

    """PBR_libFPD_predict function for facial 3D pose estimation

    Note: libFPD Wrapper API

    Arguments: 
        cv_img (opencv image) : image to detect facial 3D pose
        roi_box : face bounding box [sx, sy, ex, ey]
        cam_mtx : 3x3 input image's intrinsic matrix  
    Returns:
        predicted_results(list) : {'lm2D', 'rmtx_ortho'}

    """

    model_path='./libFPD/weights/orthof3d_resnet50.pth'
    fpose=Face3DPose(model_path)    

    return fpose.predict_persp(cv_img, roi_box, cam_mtx)
