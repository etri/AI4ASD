import csv
from pytube import YouTube
from pytube.exceptions import VideoPrivate, ExtractError, MembersOnly, PytubeError, VideoUnavailable

import torch
import argparse
import os
import pandas as pd
import cv2, tqdm
import numpy as np
from PIL import Image
import subprocess
from moviepy.editor import *

input_csv = '/home/ych/data/countix_backup/countix_test.csv'
video_dir = '/home/ych/data/countix_new/countix_test/'
gt_dir = '/home/ych/data/countix_new/countix_test_per_frame_periodicity/'

links_df = pd.read_csv(input_csv)
TOTAL_VIDEOS = links_df.shape[0]

file_list = []

for count, row in links_df.iterrows():

    filename = row['video_id']
    time_start = row['kinetics_start']
    time_end = row['kinetics_end']
    time_start_rep = row['repetition_start']
    time_end_rep = row['repetition_end']

    time_start_rep = (time_start_rep - time_start)
    time_end_rep = (time_end_rep - time_start)
    file_list.append({'filename':filename, 'time_start':time_start, 'time_start_rep':time_start_rep, 'time_end_rep':time_end_rep})

video_list = []
video_list += [f for f in sorted(os.listdir(video_dir)) if f.split('.')[-1] == 'mp4']

for idx in (range(len(video_list))):

    rgbpath = video_list[idx]
    rgb_name = rgbpath.split('.')

    video_name = os.path.join(video_dir, rgbpath)

    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    np_perframe_period = np.zeros(num_frames)

    for i in range(len(file_list)):

        files_list_name = file_list[i]['filename'] + '_' + str(file_list[i]['time_start']) + '_10'

        if rgb_name[0] == files_list_name:
            for j in range(int(file_list[i]['time_start_rep'] * fps + 0.5),
                           min(num_frames - 1, int(file_list[i]['time_end_rep'] * fps + 0.5))):
                np_perframe_period[j] = 1

    outfile_name = os.path.join(gt_dir, rgb_name[0])
    np.save(outfile_name, np_perframe_period)


    # draw color bar
    # color_bar = np.full((20, num_frames, 3), (0, 0, 255), dtype=np.uint8)
    # color_bar[:, np.where(np_perframe_period == 1), 0] = 0
    # color_bar[:, np.where(np_perframe_period == 1), 1] = 255
    # color_bar[:, np.where(np_perframe_period == 1), 2] = 0
    #
    # for fr_idx in range(num_frames):
    #     ret, frame = cap.read()
    #
    #     color_bar_copy = color_bar.copy()
    #     color_bar_copy[:, fr_idx, 0] = 255
    #     color_bar_copy[:, fr_idx, 1] = 0
    #     color_bar_copy[:, fr_idx, 2] = 0
    #
    #     frame = cv2.putText(frame, str(np_perframe_period[fr_idx]), (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
    #                         2.0, (0, 0, 255), 3, cv2.LINE_AA)
    #
    #     cv2.imshow('frame', frame)
    #     cv2.imshow('color_bar', color_bar_copy)
    #     k = cv2.waitKey(33)
    #
    #     if k == 27:  # esc key
    #         break

    cap.release()
