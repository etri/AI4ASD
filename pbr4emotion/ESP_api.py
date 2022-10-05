""" 
   * Source: ESP_api.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 21 June 2021, ETRI
   * Copyright 2022. ETRI all rights reserved. 

"""

import cv2

from libFD.api import PBR_libFD_predict
from libFD.api import PBR_libFPE_predict
from libFER.api import PBR_libFER_predict


def PBR_ESP_predict(cv_img):

    """main function

    Note: main function to predict facial expression 

    """   
    
    face_regions = PBR_libFD_predict(cv_img)
    facial_landmarks = PBR_libFPE_predict(cv_img)

    start_x, start_y, end_x, end_y, _ = face_regions[0]

    face_img = cv_img[int(start_y):int(end_y), int(start_x):int(end_x)]

    ret = PBR_libFER_predict(face_img)

    return ret



