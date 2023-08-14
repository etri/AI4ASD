"""
   * Source: demop_test.py
   * License: PBR License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 3 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import warnings
import argparse
from RRB_apis import PBR_RRB_LA, PBR_RRB_RA
from RRBNet.RepDetectModel import RepDetectNet_3D_base_s2
from RRBNet.RepClsModel import VideoSwinTransformerModel
from tqdm import tqdm
from utility import *
from data.data_load_demo import get_dataloaders, NormalizeLen
from collections import defaultdict

warnings.filterwarnings(action='ignore')
eps = 1e-7


def parse_args():

    """ parse_args function

    Note: function for user parameter setting

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='Batch Size.', type=int, default=1)
    parser.add_argument('--j', dest='num_workers', type=int, help='Dataloader CPUS', default=0)

    # User parameters (thresholds)
    parser.add_argument('--data_choice', type=str, help='data choice', default='SSBD', choices=['SSBD', 'ESBD'])

    # Parameters (RepDetectNet + RepNet)
    parser.add_argument('--vid_len_th', type=int, default=900)  #
    parser.add_argument('--div_seq_len', type=int, default=900)  #

    parser.add_argument('--consecutive_len_th', type=int, default=30)  # 30
    parser.add_argument('--method', type=int, default=2, choices=[1, 2])

    # Visualize
    parser.add_argument('--viz_scale', type=int, default=1)  # 2
    parser.add_argument('--visualize_result', type=bool, default=True)

    return parser


def PBR_RRB_detect(rgb):

    """High-level API for restricted & repetitive behavior detection

    Arguments:
        rgb (opencv image) : input video data

    Returns:
        vid_level_pred : video-level action prediction
        result_list (list) : [{'start_idx', 'end_idx', 'action_preds', 'prob'}, ..., {'preds_squeeze'}]

    """

    preds_squeeze = PBR_RRB_LA(rgb, args, model_RepDetector)

    ### smoothing results
    preds_squeeze = preds_squeeze.astype(np.uint8)
    preds_squeeze = cv2.medianBlur(preds_squeeze, 9)

    positions = defaultdict(set)
    for index, value in enumerate(preds_squeeze[:, 0]):
        positions[value].add(index)

    periodic_segment_list = ranges(positions[1])

    consecutive_len_max = 0

    for i in range(len(periodic_segment_list)):
        consecutive_len = periodic_segment_list[i][1] - periodic_segment_list[i][0]
        if consecutive_len >= consecutive_len_max:
            consecutive_len_max = consecutive_len

    consecutive_len_th = args.consecutive_len_th
    if consecutive_len_max < consecutive_len_th:
        consecutive_len_th = consecutive_len_max

    output_softmax_list = []
    result_list = []

    # loop for action classification in each proposed temporal segment
    for i in range(len(periodic_segment_list)):

        start_idx = periodic_segment_list[i][0]
        end_idx = periodic_segment_list[i][1]

        consecutive_len = end_idx - start_idx
        if consecutive_len >= consecutive_len_th:
            pass
        else:
            continue

        rgb_sub = rgb[:, start_idx:end_idx, :, :, :]
        rgb_sub = NormalizeLen(rgb_sub)  # 224, 224, 64

        output_rgb = PBR_RRB_RA(rgb_sub, model_RePNet)
        seg_level_pred = torch.argmax(output_rgb, dim=-1)
        seg_level_pred = seg_level_pred.cpu().detach().numpy()

        ## segment probability
        output_softmax = softmax(output_rgb.cpu().numpy())
        output_softmax_cls_preds = output_softmax[0][seg_level_pred]
        output_softmax_list.append(output_softmax)

        result_list.append({'start_idx': start_idx, 'end_idx': end_idx, 'action_preds': seg_level_pred,
                            'prob': output_softmax_cls_preds})

    result_list.append({'preds_squeeze':preds_squeeze})

    prob_aggre = 0
    for i in range(len(output_softmax_list)):
        prob_aggre += output_softmax_list[i]

    vid_level_pred = np.argmax(prob_aggre)

    return vid_level_pred, result_list


if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    args.datadir = os.path.join('data/video', args.data_choice)

    print("The configuration of this run is:")
    print(args, end='\n\n')

    torch.set_grad_enabled(False)
    cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    model_RepDetector = RepDetectNet_3D_base_s2()
    state_dict = torch.load('checkpoints/RRB_LA_Net_tr_countix.checkpoint')
    model_RepDetector.load_state_dict(state_dict)
    model_RepDetector.cuda()
    model_RepDetector.train(True)
    print('Loading RepDetector model done...')

    model_RePNet = VideoSwinTransformerModel(mode='demo')

    if args.data_choice == 'SSBD':
        action_cls = ['ArmFlapping', 'Spinning', 'HeadBanging']
        state_dict = torch.load('checkpoints/RRB_RA_Net_tr_ESBD_parsing.checkpoint')

    elif args.data_choice == 'ESBD':
        action_cls = ['ArmFlapping', 'Spinning', 'HeadBanging']
        state_dict = torch.load('checkpoints/RRB_RA_Net_tr_SSBD_parsing.checkpoint')

    model_RePNet.load_state_dict(state_dict)
    model_RePNet.cuda()
    model_RePNet.eval()
    print('Loading RepNet model done...')

    dataloaders = get_dataloaders(args)

    with torch.no_grad():

        for iter_idx, data in enumerate(tqdm(dataloaders['test'])):

            rgb, gt, label = [data[n].to(device) for n in ['rgb', 'gt', 'label']]
            gt = gt.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            rgb_path = data['rgbpath'][0]
            print('analyzing video: {}'.format(rgb_path))

            vid_level_pred, result_list = PBR_RRB_detect(rgb)

            print('video_level_cls_pred:', action_cls[vid_level_pred])
            print('video_level_cls_label:', action_cls[label[0]])

            if args.visualize_result == True:
                visualize_result(rgb_path, gt, args, result_list)

