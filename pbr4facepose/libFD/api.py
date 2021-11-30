""" 
   * Source: libFPD.api.py
   * License: PBR License (Dual License)
   * Modified by Howon Kim <hw_kim@etri.re.kr>
   * Date: 15 Nov 2021, ETRI

"""

import numpy as np

from libFD.arg_parser_fd import argument_parser_fd

from libFD.retinaface.detector import RetinaFaceDetector

args = argument_parser_fd()


def PBR_libFD_predict(cv_img):

    """PBR_libFD_predict function

    Note: libFD Wrapper API

    Arguments: 
        cv_img (opencv image) : image to detect faces

    Returns:
        face_regions(list of list) : [[left, top, right, bottom], ...]

    """
    # init face detector 
    detector = RetinaFaceDetector() 
        
    # infer face regions
    face_regions, facial_points = detector.detect_faces(cv_img)

    # refine face regions for face headpose detection
    for face_id in range(face_regions.shape[0]):
        roi_box = face_regions[face_id]
        
        width  = roi_box[2] - roi_box[0]
        height = roi_box[3] - roi_box[1]
        if height>width:
            length=height
        else:
            length=width
            
        length_mtf =(int)(length*0.5*1.15)     
        pos_x = int((roi_box[2] + roi_box[0])*0.5)    
        pos_y = int((roi_box[3] + roi_box[1])*0.5)
        face_center = np.array([pos_x, pos_y], dtype=np.int32)            
        face_regions[face_id][0]=face_center[0]-length_mtf
        face_regions[face_id][2]=face_center[0]+length_mtf
        face_regions[face_id][1]=face_center[1]-length_mtf
        face_regions[face_id][3]=face_center[1]+length_mtf

    return face_regions


