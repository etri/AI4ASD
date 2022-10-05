""" 
   * Source: libFER.api.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI
   * Copyright 2022. ETRI all rights reserved. 

"""

import libFER.cda_trainer as cda_trainer
import libFER.cda_predictor as cda_predictor


def PBR_libFER_train():

    """PBR_libFER_train function

    Note: libFER Wrapper API for training

    Arguments: 

    Returns:

    """

    cda_trainer.cda_train()


def PBR_libFER_predict(cv_img):

    """PBR_libFER_predict function for predicting

    Note: libFER Wrapper API

    Arguments: 
        cv_img (opencv image) : image to detect faces

    Returns:
        predicted_results(list) : ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

    """


    return cda_predictor.cda_predict(cv_img)
