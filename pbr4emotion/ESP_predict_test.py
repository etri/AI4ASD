""" 
   * Source: ESP_predict_test.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 20 Nov. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved. 

"""

import cv2
from ESP_api import PBR_ESP_predict

label_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']


def main():

    """main function

    Note: main function for a facial expression recognition task

    """
    
    cv_img = cv2.imread('./libFER/fer_test.jpg')

    ret = PBR_ESP_predict(cv_img)
    print(label_names[int(ret)])


if __name__=='__main__':
    main()