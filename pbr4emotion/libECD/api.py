"""
                                      
   * Source: api.py
   * License: PBR License (Dual License)
   * Created by ByungOk Han on 2022-01-27
   * Modified by ByungOk Han <byungok.han@etri.re.kr> on 2022-09-30
   * Copyright 2022. ETRI all rights reserved. 
                                       
"""

# -*- coding: utf-8 -*- 


import os
import argparse

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
import ruptures as rpt
from scipy.signal import savgol_filter
import timm.models 

from libFD.retinaface.detector import RetinaFaceDetector
from libECD.model_mtl import MTL_Baseline
from libECD.online_cpd_algorithms import detect_change_points_bayesian, detect_change_points_gradient
from libECD.offline_cpd_algorithms import detect_change_points_pelt, \
                                    detect_change_points_binseg, \
                                    detect_change_points_window, \
                                    detect_change_points_dynp, \
                                    detect_change_points_bottomup, \
                                    detect_change_points_kernel
from libECD.smoothing_filters import savgol, bilateral, gaussian
import csv
import time


parser = argparse.ArgumentParser(description = 'Evaluation of Emotion Change Detector')
parser.add_argument('--mter_model', default='resnet18', type=str, metavar='Model',
                    help='model type')
parser.add_argument('--initial-checkpoint', default='./resnet18_ckpt.tar', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')  
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('--video_filename', default='./test/test.avi', type=str, metavar='PATH',
                    help='path to root directory of videos')
parser.add_argument('--signal_type', default='all', type=str, metavar='signal type',
                    help='type of emotion signal')
parser.add_argument('--cpd_algorithm', default = 'bottomup', type=str, metavar = 'algo',
                    help='algorithm type of change point detection')
parser.add_argument('--cpd_pen', default=None, type=float, metavar='f',
                    help='penalty value for change point detection algorithms')
parser.add_argument('--cpd_n_bkps', default = 1 , type=int, metavar = 'i',
                    help = 'number of breakpoints parameter for change point detection algorithm')
parser.add_argument('--cpd_cost_f', default = 'rbf', type=str, metavar = 'param',
                    help = 'Cost function for changepoint detection')
parser.add_argument('--cpd_param', default = None , type=int, metavar = 'i',
                    help = 'run length of bayesian, window length of window')
parser.add_argument('--noise_removal_method', default = 'savgol', type=str, metavar = 'algo',
                    help='algorithm type of noise removal')
parser.add_argument('--smooth_window_length', default=51, type=int, metavar='i',
                    help='window length for smoothing filter')
parser.add_argument('--smooth_polyorder', default=3, type=int, metavar='i',
                    help='polynormial order for smoothing filter')
parser.add_argument('--result_filename', default='./output/result.txt', type=str, metavar='filename',
                    help='filename of recognition result')
parser.add_argument('--image_size', default=[100,100], nargs='+', type=int, metavar='SIZE',
                    help='size of image resized')
parser.add_argument('--visualize', dest='visualize', action='store_true',
                    help='visualization of emotion signals')
parser.add_argument('--signal_cache', action='store_true',
                    help='if this option is on, restore saved signal from file.')
parser.add_argument('--check_all_changepoints', action='store_true',
                    help='if this option is on, detection rate is calculated based on all detected points')
parser.add_argument('--eval_time', default=0.2, type=float, metavar='f',
                    help='penalty value for change point detection algorithms')
parser.add_argument('--video_length', default=0.0, type=float, metavar='f',
                    help='video length control paramter. if 0.1 is used, video length is 0.2')
parser.add_argument('--no_softmax', action='store_true',
                    help='softmax remove')


args = parser.parse_args()


def PBR_ESP_detect_change(video_filename):

    """main function

    Note: main function for ECD task

    """
    
    # load a face detector
    detector = RetinaFaceDetector()   

    # load a v/a model
    fe_model = timm.models.create_model(
                args.mter_model,
                pretrained=False,
                checkpoint_path="",
                num_classes=512,
                scriptable=False)
    model = MTL_Baseline(feature_model = fe_model, 
                         feature_dim = 512,  
                         num_emo_classes = 7)
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if args.initial_checkpoint:
        if os.path.isfile(args.initial_checkpoint):
            print("=> loading checkpoint '{}'".format(args.initial_checkpoint))
            checkpoint = torch.load(args.initial_checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
            print('=> loaded checkpoint "{}" (epoch {})'
                  .format(args.initial_checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.initial_checkpoint))


    dst_split_dir = args.result_filename.split('/')
    dst_dir = ''
    for dir_name in dst_split_dir:
        if dir_name != dst_split_dir[-1]:
            dst_dir=dst_dir + dir_name + '/'
            print(dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    cache_dir = './signal_cache/' + \
        'YouASD'+ '/' + \
        args.initial_checkpoint[args.initial_checkpoint.rfind('/'):] + '/'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    nEvalVideo = 0
    nHit = 0
    nCP = 0
    nFrames = 0
    all_info = []
    overall_cpd_time = 0.0

    # load a video file
    video_path = video_filename
    print('[Video File] ', video_path)

    if args.no_softmax == False:
        signal_filename = cache_dir + str(nEvalVideo) + '_signal.npz'
    else:
        signal_filename = cache_dir + str(nEvalVideo) + '_signal_no_soft.npz'

    if args.signal_cache == True:

        raw_signal = np.load(signal_filename)
                    
        emo = raw_signal['emo']
        val = raw_signal['val']
        aro = raw_signal['aro']
        time_stamp_raw = raw_signal['time']

    else:
        emo, val, aro, time_stamp_raw = PBR_ESP_track_signal(video_path, detector, model)
        np.savez(signal_filename,
            emo = emo,
            val = val,
            aro = aro,
            time = time_stamp_raw)

    nFrames = nFrames + len(time_stamp_raw)      
            
    emotion_history = np.empty((0, 7))
    valence_history = np.empty((0, 1))
    arousal_history = np.empty((0, 1))

    time_stamps = []

    for i in range(len(time_stamp_raw)):
        emotion_history = np.append(emotion_history, np.reshape(emo[i], (1, 7)), axis=0)
        valence_history = np.append(valence_history, np.reshape(val[i], (1, 1)), axis=0)
        arousal_history = np.append(arousal_history, np.reshape(aro[i], (1, 1)), axis=0)
        time_stamps.append(time_stamp_raw[i])

    # smoothing   
    if args.noise_removal_method == 'savgol':   
        for i in range(emotion_history.shape[1]):
            emotion_history[:, i] = savgol(emotion_history[:, i], 
                                            args.smooth_window_length, 
                                            args.smooth_polyorder).flatten()
        valence_history = savgol(valence_history,
                                    args.smooth_window_length, 
                                    args.smooth_polyorder)
        arousal_history = savgol(arousal_history,
                                    args.smooth_window_length, 
                                    args.smooth_polyorder) 
    elif args.noise_removal_method == 'bilateral':
        for i in range(emotion_history.shape[1]):
            emotion_history[:, i] = bilateral(emotion_history[:, i], 
                                                args.smooth_window_length, 
                                                args.smooth_polyorder).flatten()
        valence_history = bilateral(valence_history, 
                                    args.smooth_window_length, 
                                    args.smooth_polyorder)
        arousal_history = bilateral(arousal_history, 
                                    args.smooth_window_length, 
                                    args.smooth_polyorder)
    elif args.noise_removal_method == 'gaussian':
        for i in range(emotion_history.shape[1]):
            emotion_history[:, i] = bilateral(emotion_history[:, i], 
                                        args.smooth_window_length, 
                                        args.smooth_polyorder).flatten()
        valence_history = gaussian(valence_history, 
                                    args.smooth_window_length, 
                                    args.smooth_polyorder)
        arousal_history = gaussian(arousal_history, 
                                    args.smooth_window_length, 
                                    args.smooth_polyorder)
    elif args.noise_removal_method == 'no':
        print("No noise filtering method is appled")

    if args.signal_type == 'all':
        signal = np.append(emotion_history, valence_history, axis=1)
        signal = np.append(signal, arousal_history, axis=1)
    elif args.signal_type == 'va':
        signal = np.append(valence_history, arousal_history, axis=1)
    elif args.signal_type == 'valence':
        signal = valence_history
    elif args.signal_type == 'arousal':
        signal = arousal_history
    elif args.signal_type == 'emotion':
        signal = emotion_history
    elif args.signal_type == 'happiness':                
        signal = emotion_history[:, 3]     
        signal = signal.reshape(signal.shape[0], 1)
    elif args.signal_type == 'pred_emotion':
        signal = np.max(emotion_history, axis=1)

    start_cpd_time = time.time()
    if args.cpd_algorithm == 'gradient':
        results, indices = detect_change_points_gradient(
            signal,                                                                                    
            time_stamps, 
            pen=args.cpd_pen, 
            n_bkps=args.cpd_n_bkps)
    elif args.cpd_algorithm == 'pelt':
        results, indices = detect_change_points_pelt(
            signal, 
            time_stamps, 
            pen=args.cpd_pen,
            model=args.cpd_cost_f)
    elif args.cpd_algorithm == 'binseg':
        results, indices = detect_change_points_binseg(
            signal, 
            time_stamps, 
            pen=args.cpd_pen, 
            n_bkps=args.cpd_n_bkps,
            model=args.cpd_cost_f)
    elif args.cpd_algorithm == 'window':
        results, indices = detect_change_points_window(
            signal, 
            time_stamps, 
            pen=args.cpd_pen, 
            n_bkps=args.cpd_n_bkps, 
            win_width=args.cpd_param,
            model=args.cpd_cost_f)
    elif args.cpd_algorithm == 'dynp':
        results, indices = detect_change_points_dynp(
            signal, 
            time_stamps, 
            n_bkps=args.cpd_n_bkps,
            model=args.cpd_cost_f)
    elif args.cpd_algorithm == 'bottomup':
        results, indices = detect_change_points_bottomup(
            signal, 
            time_stamps, 
            pen=args.cpd_pen,
            n_bkps=args.cpd_n_bkps,
            model=args.cpd_cost_f)
    elif args.cpd_algorithm == 'kernelcpd':
        results, indices = detect_change_points_kernel(
            signal, 
            time_stamps, 
            pen=args.cpd_pen, 
            n_bkps=args.cpd_n_bkps,
            model=args.cpd_cost_f)
    elif args.cpd_algorithm == 'bayesian':
        results, indices = detect_change_points_bayesian(
            signal, 
            time_stamps, 
            run_length=args.cpd_param,
            thresh = args.cpd_pen)
    cpd_time = time.time() - start_cpd_time
    overall_cpd_time += cpd_time
        
    if time_stamps[-1] in results:
        results = results[:-1]
        indices = indices[:-1]

    #print('## [Detected change points] ', results)                  

    if args.visualize:
        # for visualization added on 2022-02-14           
        plt.figure(figsize=(20,10))                      
        line_obj = plt.plot(time_stamps, signal)
        plt.legend(labels=np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8']))  
        plt.axvline(x=results[0], color = 'r', lw=1, ls='--')    
        plt.show()

    return results, indices


def adjust_region_in_image(img, left, top, right, bottom):

    """adjust_region_in_image function

    Note: adjust_region_in_image function for ECD task

    """

    ret = 1
    (h, w) = img.shape[:2]
    if left < 0:
        left = 0
        ret = -1
    if top < 0:
        top = 0
        ret = -1
    if right >= w:
        right = w-1
        ret = -1
    if bottom >= h:
        bottom = h-1
        ret = -1

    return ret, (left, top, right, bottom)                


def PBR_ESP_track_signal(video_path, face_detector, model):

    """process_video function

    Note: process_video function for ECD task

    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dt = 1.0 / fps    

    model.eval()

    emotion_history = np.empty((0, 7))
    valence_history = np.empty((0, 1))
    arousal_history = np.empty((0, 1))
    frame_num = 0
    time_stamps = []

    while(True):       

        ret, frame = cap.read()
        time_stamp = dt * frame_num
        if ret == True :

            # detect face
            face_regions, facial_points = face_detector.detect_faces(frame)
            selected_region = []
            
            # confidence check
            for face_region in face_regions:
                if face_region[4] > 0.9: 
                    selected_region.append(face_region)
           
            # select a face closed to the center        
            if len(selected_region) == 1: 
                region = selected_region[0]
            elif len(selected_region) >= 2:                
                face_centers = []
                for face_region in selected_region:
                    face_centers.append(((face_region[0] + face_region[2])/2, (face_region[1] + face_region[3])/2))
                face_dist2center = []
                for (x, y) in face_centers:
                    face_dist2center.append((x-width/2)*(x-width/2) + (y-height/2)*(y-height/2))
                region = selected_region[face_dist2center.index(max(face_dist2center))]
            else:
                frame_num = frame_num + 1
                continue

            start_x, start_y, end_x, end_y, conf = [int(i) for i in region]
            ret, (start_x, start_y, end_x, end_y) = adjust_region_in_image(frame, start_x, start_y, end_x, end_y)

            #   v/a prediction
            face_img = frame[start_y:end_y, start_x:end_x]              
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_img)

            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])                   
            transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.ToTensor(),
                normalize])
            face_tensor = transform(face_pil)
            face_tensor = face_tensor.view(1, 3, args.image_size[0], args.image_size[1])

            if args.cuda:
                face_tensor = face_tensor.cuda()
                    
            pred_emotion, valence, arousal = model(face_tensor)
            if args.no_softmax == False:
                pred_emotion = torch.nn.functional.softmax(pred_emotion, dim=-1)

            emotion_history = np.append(emotion_history, pred_emotion.cpu().detach().numpy(), axis=0)
            valence_history = np.append(valence_history, valence.cpu().detach().numpy(), axis=0)
            arousal_history = np.append(arousal_history, arousal.cpu().detach().numpy(), axis=0)                  

            time_stamps.append(time_stamp)         
            frame_num = frame_num + 1

        else: 
            break
    
    return emotion_history, valence_history, arousal_history, time_stamps
   

if __name__=='__main__':
    main()