""" 
   * Source: libFER_predict_test.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI

"""

import cv2
from libFER.api import PBR_libFER_predict

label_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']


def main():

    """main function

    Note: main function for a facial expression recognition task

    """
    
    cv_img = cv2.imread('./libFER/fer_test.jpg')

    ret = PBR_libFER_predict(cv_img)
    print(label_names[int(ret)])


if __name__=='__main__':
    main()


