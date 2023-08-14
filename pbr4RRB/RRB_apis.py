"""
   * Source: RRB_apis.py
   * License: PBR License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 3 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""

import torch

def PBR_RRB_RA(rgb, model_RePNet):

    """Low-level API for recognizing(classifying) the detailed behavior types observed in the video segments

    Arguments:
        rgb (opencv image) : input video data

    Returns:
        vid_level_pred : video-level action prediction
        result_list (list) : [{'start_idx', 'end_idx', 'action_preds', 'prob'}, ..., {'preds_squeeze'}]

    """

    output_rgb = model_RePNet(rgb)

    return output_rgb


def PBR_RRB_LA(rgb, args, model_RepDetector):

    """Low-level API for temporal localization of repetitive actions from long untrimmed video

   Arguments:
       rgb (opencv image) : input video data
       args : argument
       model_RepDetector : model for temporal localization of repetitive actions

   Returns:
       preds_squeeze : per-frame periodicity (binary value of periodic or not)

   """

    video_len = rgb.shape[1]
    vid_len_th = args.vid_len_th

    if video_len <= vid_len_th and video_len >= 30:

        output = model_RepDetector(rgb)
        _, preds = torch.max(output, 1)

        ## Method1
        if args.method == 1:
            preds_squeeze, _ = torch.max(preds, dim=-1)
            preds_squeeze = torch.squeeze(preds_squeeze, dim=0)

        ## Method2
        elif args.method == 2:
            preds_squeeze = torch.squeeze(preds, dim=0)
            preds_squeeze = torch.diagonal(preds_squeeze, 0)

    else:
        seg_len = args.div_seq_len
        loop_num = int(video_len / seg_len + 0.5)
        preds_squeeze = torch.zeros(video_len)

        for i in range(loop_num):
            print("current loop:{} / total loop:{}".format(i + 1, loop_num))

            # Forward
            if i == loop_num - 1:
                rgb_sub = rgb[:, (i) * seg_len:, :, :, :]
            else:
                rgb_sub = rgb[:, (i) * seg_len:(i + 1) * seg_len, :, :, :]

            output_sub = model_RepDetector(rgb_sub)
            _, preds_sub = torch.max(output_sub, 1)

            # Method1
            if args.method == 1:
                preds_squeeze_sub, _ = torch.max(preds_sub, dim=-1)
                preds_squeeze_sub = torch.squeeze(preds_squeeze_sub, dim=0)

            # Method2
            elif args.method == 2:
                preds_squeeze_sub = torch.squeeze(preds_sub, dim=0)
                preds_squeeze_sub = torch.diagonal(preds_squeeze_sub, 0)

            if i == loop_num - 1:
                preds_squeeze[(i) * seg_len:] = preds_squeeze_sub

            else:
                preds_squeeze[(i) * seg_len:(i + 1) * seg_len] = preds_squeeze_sub

    preds_squeeze = preds_squeeze.cpu().numpy()

    return preds_squeeze
