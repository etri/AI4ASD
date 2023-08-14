"""
   * Source: utility.py
   * License: PBR License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 3 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""

import cv2
import numpy as np
import math
import torch
from torchvision import transforms

font_color = [(0,0,255), (0,255,0)]
action_cls = ['ArmFlapping', 'Spinning', 'HeadBanging']


def softmax(x):

    """softmax function

    Note: function for softmax operation

    """

    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def ranges(nums):

    """ranges function

    Note: function for temporal segment selection

    """

    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def optimal_font_dims(img, font_scale = 2e-3, thickness_scale = 2e-3):

    """optimal_font_dims function

    Note: function for selecting optimal font scale and thickness

    """

    h, w, _ = img.shape
    font_scale = min(w, h) * font_scale
    thickness = math.ceil(min(w, h) * thickness_scale)
    return font_scale, thickness


def visualize_result(path, gt, args, result_list):

    """visualize_result function

    Note: function for visualizing inference result(temporal localization + action classification)

    """

    with torch.no_grad():

        cap = cv2.VideoCapture(path)
        num_frames = gt.shape[1]

        # Define the codec and create VideoWriter object
        global video_num_cnt

        for fr_idx in range(num_frames):

            flag, image_bgr = cap.read()
            if not flag:
                break

            if fr_idx == 0:

                color_bar_gt = np.full((20, num_frames, 3), (0, 0, 255), dtype=np.uint8)
                color_bar_gt[:, np.where(gt == 1)[1], 0] = 0
                color_bar_gt[:, np.where(gt == 1)[1], 1] = 255
                color_bar_gt[:, np.where(gt == 1)[1], 2] = 0

                color_bar_pred = np.full((20, num_frames, 3), (0, 0, 255), dtype=np.uint8)
                color_bar_pred[:, np.where(result_list[-1]['preds_squeeze'] == 1), 0] = 0
                color_bar_pred[:, np.where(result_list[-1]['preds_squeeze'] == 1), 1] = 255
                color_bar_pred[:, np.where(result_list[-1]['preds_squeeze'] == 1), 2] = 0

                color_bar_paper = np.full((20, num_frames, 3), (255, 255, 255), dtype=np.uint8)
                color_bar_paper2 = np.full((20, num_frames, 3), (255, 255, 255), dtype=np.uint8)

            color_bar_gt_copy = color_bar_gt.copy()
            color_bar_pred_copy = color_bar_pred.copy()

            if fr_idx > 2 and fr_idx<num_frames-3:
                for i in [-3, -2, -1, 0, 1, 2, 3]:
                    color_bar_gt_copy[:, fr_idx + i, 0] = 255
                    color_bar_gt_copy[:, fr_idx + i, 1] = 0
                    color_bar_gt_copy[:, fr_idx + i, 2] = 0
                    color_bar_pred_copy[:, fr_idx + i, 0] = 255
                    color_bar_pred_copy[:, fr_idx + i, 1] = 0
                    color_bar_pred_copy[:, fr_idx + i, 2] = 0

            color_bar_cat = cv2.vconcat([color_bar_paper, color_bar_pred_copy])
            color_bar_cat = cv2.vconcat([color_bar_cat, color_bar_paper2])
            color_bar_cat = cv2.vconcat([color_bar_cat, color_bar_gt_copy])
            color_bar_cat = cv2.resize(color_bar_cat,
                                       dsize=(250 * args.viz_scale * 3,
                                       int(250 * args.viz_scale * 3 * 20 / num_frames*4)),
                                       interpolation=cv2.INTER_NEAREST)

            frame_resize = cv2.resize(image_bgr,
                                      dsize=(color_bar_cat.shape[1],
                                      int(image_bgr.shape[0] / image_bgr.shape[1] * color_bar_cat.shape[1])),
                                      interpolation=cv2.INTER_LINEAR)

            y_anchor, x_anchor, _ = frame_resize.shape

            if len(result_list) != 0:
                for i in range(len(result_list)-1):
                    if fr_idx >= result_list[i]['start_idx'] and fr_idx <= result_list[i]['end_idx']:

                        action_preds = action_cls[result_list[i]['action_preds'][0]]

                        font_scale, thickness = optimal_font_dims(frame_resize)
                        text = action_preds + "(" + str(int(result_list[i]['prob'] * 1000) / 10) + '%)'
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,thickness)
                        text_w, text_h = text_size
                        pos = (int(x_anchor * 0.05), int(y_anchor * 0.7))
                        cv2.rectangle(frame_resize, pos, (pos[0] + text_w, pos[1] + text_h+10), (0, 255, 0), -1)
                        cv2.putText(frame_resize, text, (pos[0], pos[1] + text_h + 2 - 1),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

            result = cv2.vconcat([frame_resize, color_bar_cat])

            cv2.imshow('result', result)
            k = cv2.waitKey(1)
            if k == 27:  # esc key
                break

            torch.cuda.empty_cache()

        cap.release()
