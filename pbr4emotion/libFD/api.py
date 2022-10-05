""" 
   * Source: libFD.api.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI
   * Copyright 2022. ETRI all rights reserved. 

"""

from libFD.arg_parser_fd import argument_parser_fd

from libFD.mtcnn.detector import MtcnnDetector
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

    global args
    # load a face detector
    if args.det_method == 'mtcnn':
        detector = MtcnnDetector()
    elif args.det_method == 'retina':
        detector = RetinaFaceDetector() 
        
    face_regions, facial_points = detector.detect_faces(cv_img)

    return face_regions


def PBR_libFPE_predict(cv_img):

    """PBR_libFPE_predict function

    Note: libFPE Wrapper API

    Arguments: 
        cv_img (opencv image) : image to detect faces

    Returns:
        facial_points(list of list) : [[x, y], ...] 

    """

    if args.det_method == 'mtcnn':
        detector = MtcnnDetector()
    elif args.det_method == 'retina':
        detector = RetinaFaceDetector() 
        
    face_regions, facial_points = detector.detect_faces(cv_img)

    return facial_points



