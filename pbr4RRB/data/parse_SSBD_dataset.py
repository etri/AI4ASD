import os
import numpy as np
import cv2
from xml.etree.ElementTree import parse
from pathlib import Path

action_cls_mapping ={'armflapping':'ArmFlapping', 'spinning':'Spinning', 'headbanging':'HeadBanging', 'jumping':'Jumping', 'toyplaying':'ToyPlaying', 'etc':'etc'}
cls_idx_mapping = {'ArmFlapping':1, 'Spinning':2, 'HeadBanging':3, 'ToyPlaying':4, 'Jumping':5, 'etc':6}

file_dir = '/home/ych/data/SSBD/Dataset_revised_ych/'
gt_dir = '/home/ych/data/SSBD/Annotations_revised_ych'

gt_list = []
gt_list += [os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir)) if f.split(".")[-1] == "xml"]


for i in range(len(gt_list)):

    file_name = gt_list[i].split('/')
    file_name = file_name[-1].split('.')
    file_name = file_name[0]
    action_cls = file_name.split('_')

    video_name = file_dir + action_cls[1] + '/' + file_name + '.mp4'
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if num_frames == 0:
        continue

    np_perframe_period = np.zeros(num_frames)

    folder_name_period = file_dir + action_cls[1] + '_per_frame_periodicity_colorful'
    Path(folder_name_period).mkdir(parents=True, exist_ok=True)
    video_name_period = folder_name_period + '/' + file_name + '.npy'
    print('parsing file:', file_name)

    tree = parse(gt_list[i])
    root = tree.getroot()

    for behaviours in root.findall("behaviours"):
        time = [x.findtext("time") for x in behaviours]
        category = [x.findtext("category") for x in behaviours]

        for i in range(len(time)):

            time_split = time[i].split(':')

            if len(time_split) == 1:
                time_split = time[i].split('-')

            time_start = int(time_split[0])
            time_end = int(time_split[1])

            if time_start >= 100: # min sec -> sec
                time_start = int(time_start/100) * 60 + int(time_start%100)
            if time_end >= 100:
                time_end = int(time_end/100) * 60 + int(time_end%100)

            frame_start = int(time_start * fps + 0.5)
            frame_end = int(time_end * fps + 0.5)

            for j in range(frame_start, min(num_frames - 1, frame_end)):
                np_perframe_period[j] = cls_idx_mapping[action_cls_mapping[category[i]]]

            video_name_parsed = file_dir + action_cls_mapping[category[i]] + '_parsed' + '/' + file_name + '_'+ str(i).zfill(3) + '.mp4'
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_name_parsed, fourcc, 30, (frameWidth, frameHeight))

            num_frame_repetition = frame_end - frame_start + 1
            for fr_idx in range(num_frame_repetition):
                cap.set(1, frame_start + fr_idx)
                ret, frame = cap.read()
                out.write(frame)

            out.release()

        np.save(video_name_period, np_perframe_period)

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