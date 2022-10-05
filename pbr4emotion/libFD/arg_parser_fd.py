""" 
   * Source: libFD.arg_parser_fd.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI
   * Copyright 2022. ETRI all rights reserved. 

"""

import argparse


def argument_parser_fd():

    """argument_parser_fd function

    Note: argument_parser_fd function

    """

    parser = argparse.ArgumentParser(description = 'Face Detection')

    parser.add_argument('--det_method', default='retina', type=str, metavar='model',
                    help='face detect method(mtcnn, retina)')

    #args = parser.parse_args()
    args, _ = parser.parse_known_args()

    return args
