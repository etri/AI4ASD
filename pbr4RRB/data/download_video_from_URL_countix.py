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

input_csv = '/home/ych/data/countix_backup/countix_train.csv'
video_dir = '/home/ych/data/countix_new/countix_train/'
gt_dir = '/home/ych/data/countix_new/countix_train_per_frame_periodicity/'

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

    url_name = 'https://www.youtube.com/watch?v=' + filename
    target_file_name = video_dir + filename+ '.mp4'

    # for trimming
    start = str(time_start)
    end = str(time_end - time_start)

    input_filename = target_file_name
    output_filename = video_dir + filename + '_{}_{}'.format(start, end) + '.mp4'

    print('try downloading:', filename)
    if os.path.exists(output_filename):
        print('file already exist')
        continue
    # download from URL
    try:
        yt = YouTube(url_name)
        stream = yt.streams.get_highest_resolution()
        stream.download(filename=target_file_name)
    except (Exception):
        try:
            yt = YouTube(url_name)
            stream = yt.streams.get_highest_resolution()
            stream.download(filename=target_file_name)
        except (Exception):
            try:
                yt = YouTube(url_name) # try three times
                stream = yt.streams.get_highest_resolution()
                stream.download(filename=target_file_name)
            except (Exception):
                    print('video unavailable for some reasons')
                    continue


    clip = VideoFileClip(input_filename)
    out = clip.subclip(time_start, min(clip.end, time_end))
    out.write_videofile(output_filename, audio=False)

    print('Finish trimming: ', filename)

    os.remove(input_filename)

    ##
    cap = cv2.VideoCapture(output_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name_period = gt_dir + filename + '_{}_{}'.format(start, end) + '.npy'

    if os.path.exists(video_name_period):
        np_perframe_period = np.load(video_name_period)
    else:
        np_perframe_period = np.zeros(num_frames)

    frame_start = int(time_start_rep * fps + 0.5)
    frame_end = int(time_end_rep * fps + 0.5)

    for j in range(frame_start, min(num_frames - 1, frame_end)):
        np_perframe_period[j] = 1

    np.save(video_name_period, np_perframe_period)

    print('Processed %i out of %i' % (count + 1, TOTAL_VIDEOS))
