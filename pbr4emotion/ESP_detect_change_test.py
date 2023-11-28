""" 
   * Source: ESP_detect_change_test.py
   * License: PBR License (Dual License)
   * Created by ByungOk Han <byungok.han@etri.re.kr> on 2023-11-20
   * Copyright 2023. ETRI all rights reserved. 

"""

import cv2
from libECD.api import PBR_ESP_detect_change


def main():

    """main function

    Note: main function for ECD task

    """
    
    video_filename = './test/test.avi'

    ret, idx = PBR_ESP_detect_change(video_filename)
    print('Time for emotion change: ', ret, 'sec', '(', idx, 'th frame)')


if __name__=='__main__':
    main()


