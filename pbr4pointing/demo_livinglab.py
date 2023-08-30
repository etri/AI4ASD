"""
   * Source: demo_livinglab.py
   * License: PBR4AI License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 21 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import copy
import torchvision.transforms as transforms

from argparse import ArgumentParser
from lib.network.rtpose_vgg import get_model
from lib.config import cfg
from lib.utils.paf_to_pose import paf_to_pose_cpp
from evaluate.coco_eval import get_outputs_openpose
from pyk4a import PyK4APlayback
from tqdm import tqdm
from utility import *
from PIL import Image
from model.model import build_net

eps = 1e-7
IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])
pointing_label = ['No pointing', 'Pointing']
font_color = [(0, 0, 255), (0, 255, 0)]
m = nn.Softmax(dim=1)

tf_Resize = transforms.Resize((224, 224))
tf_ToTensor = transforms.ToTensor()
tf_Normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)


def parse_args():

    """ parse_args function

    Note: function for user parameter setting

    """

    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')

    ## visualize
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='whether to show visualizations.')

    parser.add_argument(
        '--viz_skeleton',
        default=False,
        help='whether to visualize skeleton.')

    parser.add_argument(
        '--input',
        type=str,
        default='mkv',
        help='live or saved video',
        choices=['mkv', 'kinect', 'webcam'])

    # User parameters (thresholds)
    parser.add_argument('--k_val', type=float, default=1.5)
    parser.add_argument('--box_half_len', type=int, default=70)
    parser.add_argument('--positive_persist_thres', type=int, default=2)  # 5
    parser.add_argument('--dist_thres', type=int, default=2000)
    parser.add_argument('--detect_hand', type=str, default='openpose', choices=['rcnn', 'openpose'])
    parser.add_argument('--select_person', type=str, default='min_bone', choices=['min_bone', 'nearest'])
    parser.add_argument('--showFps', type=bool, default=False)
    parser.add_argument('--viz_scale', type=int, default=1.0)
    parser.add_argument('--model_name', type=str, default='resnet50_ntu_SimSiam')
    parser.add_argument('--test_name', type=str, default=None)
    parser.add_argument('--SSL', type=str, default='SimSiam', choices=['None', 'SimSiam', 'BYOL'])
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'vit_B_32'])

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    print("The configuration of this run is:")
    print(args, end='\n\n')

    if args.input == 'mkv':
        base_dir_pos = 'living_lab_db/contents/pointing_positive/'
        folder_list_pos = os.listdir(base_dir_pos)
        input_list_pos = []
        labels_pos = []
        labels = []

        for i in range(len(folder_list_pos)):
            file_dir = base_dir_pos + folder_list_pos[i] + '/08/rec/'
            input_list_pos += [os.path.join(file_dir, f) for f in sorted(os.listdir(file_dir)) if
                               f.split(".")[-1] == "mkv"]
        labels_pos = [0] * len(input_list_pos)

        base_dir_neg = 'living_lab_db/contents/pointing_negative/'
        folder_list_neg = os.listdir(base_dir_neg)
        input_list_neg = []
        labels_neg = []

        for i in range(len(folder_list_neg)):
            file_dir = base_dir_neg + folder_list_neg[i] + '/08/rec/'
            input_list_neg += [os.path.join(file_dir, f) for f in sorted(os.listdir(file_dir)) if
                               f.split(".")[-1] == "mkv"]
        labels_neg = [1] * len(input_list_neg)

        input_list = input_list_pos + input_list_neg
        labels = labels_pos + labels_neg

        # camera configuration
        camera_param = {
            'fx': 964.9,
            'fy': 963.6,
            'cx': 1024.4,
            'cy': 779.7
        }

        fx = camera_param['fx']
        fy = camera_param['fy']
        cx = camera_param['cx']
        cy = camera_param['cy']
        BOX_HALF_LEN = args.box_half_len
        k_val = args.k_val

    # Load OpenPose
    model_oepnpose = get_model('vgg19')
    model_oepnpose.load_state_dict(torch.load('checkpoints/pose_model.pth'))
    model_oepnpose.cuda()
    model_oepnpose.float()
    model_oepnpose.eval()
    print('loading openpose done...')

    ## model(handNet) ##
    model_PointDetNet = build_net(args)

    if args.SSL == 'None':
        checkpoint = \
            torch.load('checkpoints/logs/' + args.model_name + '/model_best.checkpoint', map_location='cuda:0')
        model_PointDetNet.load_state_dict(checkpoint)

    else:
        state_dict = \
            torch.load('checkpoints/logs/' + args.model_name + '/model_best.checkpoint', map_location='cuda:0')
        model_dict = model_PointDetNet.state_dict()

        # 1. filter out unnecessary keys
        state_dict = {k: v for k, v in state_dict.items() if
                      (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(state_dict)
        model_PointDetNet.load_state_dict(model_dict)

    model_PointDetNet.cuda()
    model_PointDetNet.eval()
    print('loading PointDetNet(hand) done...')

    running_corrects = 0.0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    ndata = 0
    video_num_cnt = 0

    if args.test_name == None:
        args.test_name = args.model_name
        txt_dir = args.model_name + '/Results_test.txt'
    else:
        txt_dir = 'Results_parsing_DB_final_' + args.test_name + '.txt'

    for idx_list in tqdm(range(len(input_list))):

        text_color = (0, 0, 255)
        text_color_pointing = (0, 0, 255)

        prob_hand_list_video = []
        pred_list_video = []
        prob_pointing = 0
        pred_pointing = False
        frame_count = 0
        pos_cnt = 0
        pred_buf = 0
        prob_hand = 0
        fr_idx = 0

        if args.input == 'mkv':
            playback = PyK4APlayback(input_list[idx_list])
            playback.open()

        while True:
            try:
                last_time = time.time()

                capture = playback.get_next_capture()

                if capture.color is not None and capture.depth is not None:
                    img = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)
                    depth = capture.transformed_depth
                else:
                    continue

                img_copy = copy.deepcopy(img)
                center_pt3_2d = (0, 0)

                with torch.no_grad():

                    ## Openpose part
                    paf, heatmap, imscale = get_outputs_openpose(
                        img_copy, model_oepnpose, 'rtpose')

                    humans = paf_to_pose_cpp(heatmap, paf, cfg)

                    # generate detection result
                    image_h, image_w = img.shape[:2]
                    hand_bbox_results = []

                    dist_min = 1e20
                    baby_idx = None

                    if args.select_person == 'min_bone':
                        for i in range(len(humans)):
                            if 0 in humans[i].body_parts.keys() and 2 in humans[i].body_parts.keys() and 5 in humans[
                                i].body_parts.keys():

                                x_scale = int(humans[i].body_parts[2].x * image_w + 0.5)
                                y_scale = int(humans[i].body_parts[2].y * image_h + 0.5)
                                pt_idx_0 = transform_2d_to_3d(depth, x_scale, y_scale, camera_param)

                                x_scale = int(humans[i].body_parts[5].x * image_w + 0.5)
                                y_scale = int(humans[i].body_parts[5].y * image_h + 0.5)
                                pt_idx_1 = transform_2d_to_3d(depth, x_scale, y_scale, camera_param)

                                if pt_idx_0[2] > args.dist_thres:
                                    continue

                                dist = cal_dist(pt_idx_0, pt_idx_1)

                                if dist <= dist_min:
                                    dist_min = dist
                                    baby_idx = i

                    elif args.select_person == 'nearest':
                        for i in range(len(humans)):
                            if 0 in humans[i].body_parts.keys():

                                x_scale = int(humans[i].body_parts[0].x * image_w + 0.5)
                                y_scale = int(humans[i].body_parts[0].y * image_h + 0.5)
                                dist = depth[y_scale, x_scale]

                                if dist == 0 or dist > args.dist_thres:
                                    continue

                                if dist <= dist_min:
                                    dist_min = dist
                                    baby_idx = i

                    img = np.ascontiguousarray(img, dtype=np.uint8)

                    Coco_class_num = 18

                    for idx, human in enumerate(humans):
                        # draw point

                        if idx == baby_idx: 
                            for i in range(Coco_class_num):

                                if i not in humans[baby_idx].body_parts.keys():
                                    continue
                                if i == 4 or i == 7:

                                    pre_idx = i - 1 

                                    if pre_idx in human.body_parts.keys():

                                        body_part_pre = human.body_parts[i - 1]
                                        body_part = human.body_parts[i]

                                        # 3d bbox detection
                                        pre_x_scale = int(body_part_pre.x * image_w + 0.5)
                                        pre_y_scale = int(body_part_pre.y * image_h + 0.5)
                                        x_scale = int(body_part.x * image_w + 0.5)
                                        y_scale = int(body_part.y * image_h + 0.5)

                                        center_pt1 = transform_2d_to_3d(depth, pre_x_scale, pre_y_scale, camera_param)
                                        center_pt2 = transform_2d_to_3d(depth, x_scale, y_scale, camera_param)

                                        grad = (center_pt2[0] - center_pt1[0], center_pt2[1] - center_pt1[1],
                                                center_pt2[2] - center_pt1[2])

                                        # co-linearity
                                        center_pt3 = (center_pt1[0] + k_val * grad[0], center_pt1[1] + k_val * grad[1],
                                                      center_pt1[2] + k_val * grad[2])

                                        if center_pt1[2] <= 0 or center_pt2[2] <= 0 or center_pt3[2] <= 0:
                                            continue
                                            
                                        Xmin = center_pt3[0] - BOX_HALF_LEN
                                        Xmax = center_pt3[0] + BOX_HALF_LEN
                                        Ymin = center_pt3[1] - BOX_HALF_LEN
                                        Ymax = center_pt3[1] + BOX_HALF_LEN
                                        Zmin = center_pt3[2] - BOX_HALF_LEN
                                        Zmax = center_pt3[2] + BOX_HALF_LEN

                                        xmin = fx * Xmin / center_pt3[2] + cx
                                        xmax = fx * Xmax / center_pt3[2] + cx
                                        ymin = fy * Ymin / center_pt3[2] + cy
                                        ymax = fy * Ymax / center_pt3[2] + cy
                                        xmin = np.clip(xmin, 0, image_w)

                                        box_arr_hand = np.array([xmin, ymin, xmax, ymax, 0.99])
                                        hand_bbox_results.append({'bbox': box_arr_hand})

                    preds_list = []
                    prob_hand_list = []

                    for i in range(len(hand_bbox_results)):
                        ## Hand
                        x_min = max(int(hand_bbox_results[i]['bbox'][0]), 0)
                        y_min = max(int(hand_bbox_results[i]['bbox'][1]), 0)
                        x_max = min(int(hand_bbox_results[i]['bbox'][2]), image_w)
                        y_max = min(int(hand_bbox_results[i]['bbox'][3]), image_h)

                        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), text_color, 4)

                        hand_roi = img_copy[y_min:y_max, x_min:x_max, :]
                        if hand_roi.size == 0:
                            continue

                        hand_roi = hand_roi[:, :, ::-1]
                        hand_roi = Image.fromarray(hand_roi)
                        hand_roi = tf_Resize(hand_roi)
                        hand_roi = tf_ToTensor(hand_roi)
                        hand_roi = tf_Normalize(hand_roi)
                        hand_roi = torch.unsqueeze(hand_roi, 0)

                        ## PointDetNet model inference
                        if args.SSL == 'None':
                            outputs_hand = model_PointDetNet(hand_roi.cuda())
                        else:
                            outputs_hand, _ = model_PointDetNet(hand_roi.cuda(), hand_roi.cuda())

                        _, predictions_hand = torch.max(outputs_hand, 1)
                        pred = predictions_hand.cpu().detach().numpy()

                        prob_hand = softmax(outputs_hand.cpu().detach().numpy())
                        prob_hand = prob_hand[0][0] * 100
                     
                        preds_list.append(pred)
                        prob_hand_list.append(prob_hand)

                        if pred == 1:  # pointing gesture no
                            text_color = (0, 0, 255)

                        else:  # pointing gesture yes
                            text_color = (0, 255, 0)

                        # visualize pointing or not in each hand
                        text = pointing_label[1 - int(pred)]
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                        text_w, text_h = text_size
                        pos = (x_min, max(0,y_min - 60))
                        img = cv2.rectangle(img, pos, (pos[0] + text_w, pos[1] + text_h + 10), text_color, -1)
                        cv2.putText(img, text, (pos[0], pos[1] + text_h + 2 - 1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

                        ## alpha blending
                        img_cp = img.copy()
                        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), text_color, -1)
                        alpha = 0.1
                        img = cv2.addWeighted(img, alpha, img_cp, 1 - alpha, gamma=0)
                        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), text_color, 4)

                    prob_hand_list_video.append(prob_hand_list)
                    pred_list_video.append(preds_list)

                    if len(preds_list) == 0:
                        aggregated_pred = 0
                    else:
                        aggregated_pred = 1 - min(preds_list)

                    positive_persist_thres = args.positive_persist_thres

                    if args.positive_persist_thres == 0:
                        pred_buf = 1
                        positive_persist_thres = 1
                    if (pred_buf == 1 and int(aggregated_pred) == 1):
                        pos_cnt += 1
                        text_color_arm = (0, 0, 255)
                    if (int(aggregated_pred) == 0):
                        pos_cnt = 0
                    if (pos_cnt >= positive_persist_thres and pred_pointing == False):
                        pred_pointing = True
                        text_color_pointing = (0, 255, 0)

                        for i in range(args.positive_persist_thres + 1):
                            index = pred_list_video[-1 - i].index(0) 
                            prob_pointing += prob_hand_list_video[-1 - i][index]

                        prob_pointing /= float(args.positive_persist_thres + 1)

                    pred_buf = int(aggregated_pred)

                img = cv2.putText(img, 'pred_pointing: '  "%s" % (pred_pointing), (25, 150),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color_pointing, 2)
                img = cv2.putText(img, 'label: '  "%s" % (pointing_label[1 - labels[idx_list]]), (25, 200),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)

                if args.viz_scale != 1:
                    img = cv2.resize(img, dsize=(0, 0), fx=args.viz_scale, fy=args.viz_scale,
                                     interpolation=cv2.INTER_LINEAR)
                if args.show:
                    cv2.imshow('Image', img)
                    k = cv2.waitKey(1)

                    if k == 27:  # esc key`
                        break

                frame_count += 1

            except EOFError:
                break

        pointing_prob = None

        if pred_pointing == True:
            pred = 0
            pointing_prob = prob_pointing

        elif pred_pointing == False:
            pred = 1
            new_list = []
            prob_hand_list_video = np.array(prob_hand_list_video)
            for i in range(len(prob_hand_list_video)):
                new_list.append(np.mean(prob_hand_list_video[i]))

            new_list = [x for x in new_list if math.isnan(x) == False]
            pointing_prob = np.mean(new_list)

        fp = open(os.path.join('checkpoints/logs', txt_dir), 'a')

        if labels[idx_list] == 0:
            gt = True
        else:
            gt = False

        if gt == pred_pointing:
            correct = 1
        else:
            correct = 0

        fp.write(
            'path: {}, gt_pointing.: {}, pred_pointing: {}, correct: {}, pointing_prob: {:.1f} \n'.
                format(input_list[idx_list], gt, pred_pointing, correct, pointing_prob))
        fp.close()

        # Calculate accuracy
        if pred == labels[idx_list]:
            running_corrects += 1

            if pred == 0:
                TP = TP + 1
            else:
                TN = TN + 1
        else:
            if pred == 0:
                FP = FP + 1
            else:
                FN = FN + 1

        ndata += 1

        if args.input == 'mkv':
            playback.close()

        if args.show:
            cv2.destroyAllWindows()

    # Final accuracy
    acc = running_corrects / ndata
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f_score = 2 * (precision * recall) / (precision + recall + eps)

    print('test_Accuracy: {:.3f}, recall: {:.3f}, precision: {:.3f}, f_score: {:.3f}'
          .format(acc, recall, precision, f_score))
    print('TP: {}, FP: {}, TN: {}, FN: {} '.format(TP, FP, TN, FN))

    fp = open(os.path.join('checkpoints/logs', txt_dir), 'a')
    fp.write('ensemble_th: {} \n'.format(args.positive_persist_thres))
    fp.write(
        'model_hand: {} test_Accuracy: {:.3f}, recall:{:.3f}, precision:{:.3f}, f_socre:{:.3f}'
        ', TP: {}, FP: {}, TN: {}, FN: {}  \n'.
            format(args.model_name, acc, recall, precision, f_score, TP, FP, TN, FN))
    fp.close()
