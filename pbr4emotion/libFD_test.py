""" 
   * Source: libFD_test.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI
   * Copyright 2022. ETRI all rights reserved. 

"""

import cv2

from libFD.api import PBR_libFD_predict
from libFD.api import PBR_libFPE_predict


def main():

    """main function

    Note: main function for a face detection task

    """
    
    cv_img = cv2.imread('./libFD/fd_test.jpg')

    bbox = PBR_libFD_predict(cv_img)
    facial_landmarks = PBR_libFPE_predict(cv_img)

    cv_img = visualize(cv_img, bbox, facial_landmarks)

    cv2.imshow('test', cv_img)
    cv2.waitKey(0)


def visualize(cv_img, bbox, landmarks):

    """visualize function

    Note: visualization function using facial points and bounding boxes

    Arguments:
        cv_img (opencv image): image to visualize bounding boxes and landmarks
        bbox(list of list): [[left, top, right, bottom],...]
        landmarks(list of list): [[x, y],...]

    Returns:
        cv_img (opencv image): a visualized image

    """     

    for b in bbox:
        cv2.rectangle(cv_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)

    for p in landmarks:
        for i in range(5):
            cv2.circle(cv_img, (int(p[i]), int(p[i + 5])), 1, (0, 255, 0), -1)

    return cv_img


if __name__=='__main__':
    main()

