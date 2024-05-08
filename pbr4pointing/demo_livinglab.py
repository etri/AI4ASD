"""
   * Source: demo_livinglab.py
   * License: PBR License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 21 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torchvision.transforms as transforms
import mediapipe as mp
import onnxruntime as ort

from argparse import ArgumentParser
from tqdm import tqdm
from utility import *
from PIL import Image
from model.model import build_net
from typing import Tuple, Union
from torchvision.ops import nms

eps = 1e-7
IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])
pointing_label = ['No pointing', 'Pointing']
font_color = [(0, 0, 255), (0, 255, 0)]
m = nn.Softmax(dim=1)

tf_Resize = transforms.Resize((224, 224))
tf_ToTensor = transforms.ToTensor()
tf_Normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

# mediapipe (do not change)
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5

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

    # Parameters (yolo-world)
    parser.add_argument('--iou_threshold', help='iou_threshold for NMS', default=0.6)
    parser.add_argument('--confidence_threshold', help='confidence_threshold', default=0.2)  # 0.05

    # Parameters (mediapipe)
    parser.add_argument('--static_image_mode', type=str, default=False)
    parser.add_argument('--max_num_hands', type=int, default=6)
    parser.add_argument('--model_complexity', type=int, default=0)
    parser.add_argument('--min_detection_confidence', type=float, default=0.3)  # 0.5
    parser.add_argument('--min_tracking_confidence', type=float, default=0.5)

    # User parameters (thresholds)
    parser.add_argument('--positive_persist_thres', type=int, default=2)  # 5
    parser.add_argument('--showFps', type=bool, default=False)
    parser.add_argument('--viz_scale', type=int, default=1.0)
    parser.add_argument('--model_name', type=str, default='resnet50_ntu_SimSiam')
    parser.add_argument('--test_name', type=str, default=None)
    parser.add_argument('--SSL', type=str, default='SimSiam', choices=['None', 'SimSiam', 'BYOL'])
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'vit_B_32'])

    return parser.parse_args()


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


if __name__ == '__main__':

    args = parse_args()
    print("The configuration of this run is:")
    print(args, end='\n\n')

    base_dir_pos = 'living_lab_db/pointing_positive/'
    folder_list_pos = os.listdir(base_dir_pos)
    input_list_pos = []
    labels_pos = []
    labels = []

    for i in range(len(folder_list_pos)):
        file_dir = base_dir_pos + folder_list_pos[i]
        input_list_pos += [os.path.join(file_dir, f) for f in sorted(os.listdir(file_dir)) if
                           f.split(".")[-1] == "mp4" or f.split(".")[-1] == "avi" or f.split(".")[-1] == "mkv"]
    labels_pos = [0] * len(input_list_pos)

    base_dir_neg = 'living_lab_db/pointing_negative/'
    folder_list_neg = os.listdir(base_dir_neg)
    input_list_neg = []
    labels_neg = []

    for i in range(len(folder_list_neg)):
        file_dir = base_dir_neg + folder_list_neg[i]
        input_list_neg += [os.path.join(file_dir, f) for f in sorted(os.listdir(file_dir)) if
                           f.split(".")[-1] == "mp4" or f.split(".")[-1] == "avi" or f.split(".")[-1] == "mkv"]
    labels_neg = [1] * len(input_list_neg)

    input_list = input_list_pos + input_list_neg
    labels = labels_pos + labels_neg

    # load body detect model (yolo-world)
    model_bodycrop = ort.InferenceSession('checkpoints/child_adult_yolow.onnx', providers=['CUDAExecutionProvider'])
    print('Loading bodycrop model(yolo-world) done...')

    # load hand detect model (mediapipe)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        args.static_image_mode,
        args.max_num_hands,
        args.model_complexity,
        args.min_detection_confidence,
        args.min_tracking_confidence)

    print('loading mediapipe hand model done...')

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

        cap = cv2.VideoCapture(input_list[idx_list])  # 0, 2

        while True:
            try:
                last_time = time.time()

                flag, img = cap.read()
                if not flag:
                    break

                img_copy = img.copy()
                image_h, image_w = img.shape[:2]

                with torch.no_grad():

                    hand_bbox_results = []
                    scale = 640.0 / max(image_w, image_h)
                    new_width = int(image_w * scale)
                    new_height = int(image_h * scale)

                    # image resizing
                    resized_image = cv2.resize(img_copy, (new_width, new_height))

                    # Zero padding 
                    if new_width > new_height:
                        top = 0
                        bottom = (640 - new_height)
                        left = right = 0
                    else:
                        top = bottom = 0
                        left = right = (640 - new_width) // 2

                    # add padding to image
                    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                      value=[0, 0, 0])
                    input_data = padded_image.astype('float32') / 255

                    input_data = np.expand_dims(input_data, axis=0)
                    input_data = np.transpose(input_data, (0, 3, 1, 2))

                    ########################################################################
                    ############################# Inference ################################
                    ########################################################################
                    input_name = model_bodycrop.get_inputs()[0].name
                    output_names = [output.name for output in model_bodycrop.get_outputs()]

                    result = model_bodycrop.run(output_names, {input_name: input_data})
                    ## box coords rescale
                    result[1][0][:-1][:, [0, 2]] = result[1][0][:-1][:, [0, 2]] / (scale * image_w)
                    result[1][0][:-1][:, [1, 3]] = result[1][0][:-1][:, [1, 3]] / (scale * image_h)

                    boxes = torch.Tensor(result[1][0][:-1])
                    logits = torch.Tensor(result[2][0][:-1])
                    phrases = torch.Tensor(result[3][0][:-1])

                    ########################################################################
                    ###################### Non-Maximum Suppression #########################
                    ########################################################################
                    IOU_THRESHOLD = args.iou_threshold
                    nms_idx = nms(torch.cat((boxes[:, :2], boxes[:, 2:]), dim=1), logits,
                                  IOU_THRESHOLD).numpy().tolist()  ## yolo-world ver.
                    boxes = boxes[nms_idx]
                    logits = logits[nms_idx]
                    phrases = phrases[nms_idx]

                    ########################################################################
                    ####################### Confidence Thresholding ########################
                    ########################################################################
                    CONFIDENCE_THRESHOLD = args.confidence_threshold
                    boxes = boxes[logits > CONFIDENCE_THRESHOLD]
                    phrases = phrases[logits > CONFIDENCE_THRESHOLD]
                    logits = logits[logits > CONFIDENCE_THRESHOLD]

                    boxes = boxes * torch.Tensor([image_w, image_h, image_w, image_h])

                    # Filter logits based on phrases
                    logits_filtered = logits[phrases == 0]  ## 0: child, 1: adult

                    if logits_filtered.numel() > 0:
                        # Find the index of the maximum value in the filtered logits
                        max_idx = torch.argmax(logits_filtered)

                        # Retrieve the corresponding box using the index
                        max_box = boxes[phrases == 0][max_idx]
                        pt1 = (int(max_box[0]), int(max_box[1]))
                        pt2 = (int(max_box[2]), int(max_box[3]))

                        box_width = pt2[0] - pt1[0]
                        box_height = pt2[1] - pt1[1]


                        bbox_margin_w = int(box_width / 8.0)  # add margin to box x1.5 (4.0)
                        bbox_margin_h = 0  # int(box_height / 8.0)  # add margin to box x1.5 (4.0)

                        bbox_xmin = max(int(pt1[0] - bbox_margin_w), 0)
                        bbox_ymin = max(int(pt1[1] - bbox_margin_h), 0)
                        bbox_xmax = min(int(pt2[0] + bbox_margin_w), image_w - 1)
                        bbox_ymax = min(int(pt2[1] + bbox_margin_h), image_h - 1)

                        pt1 = (bbox_xmin, bbox_ymin)
                        pt2 = (bbox_xmax, bbox_ymax)

                        cv2.rectangle(img, pt1, pt2, (255, 0, 255), 2)
                        cv2.putText(img, 'child', (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (255, 0, 255), 2)
                        cv2.putText(img, str(int(logits_filtered[max_idx].numpy() * 1000) / 10.0),
                                    (pt1[0] + 80, pt1[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

                        img_croppped = img_copy[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax, :]

                        # To improve performance, optionally mark the image as not writeable to
                        # pass by reference.
                        img_croppped.flags.writeable = False
                        img_croppped = cv2.cvtColor(img_croppped, cv2.COLOR_BGR2RGB)

                        results = hands.process(img_croppped)

                        # Draw the hand annotations on the image.
                        img_croppped.flags.writeable = True
                        img_croppped = cv2.cvtColor(img_croppped, cv2.COLOR_RGB2BGR)

                        image_h_cropped, image_w_cropped = img_croppped.shape[:2]

                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    img_croppped,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style())

                                x_list = []
                                y_list = []
                                for idx, landmark in enumerate(hand_landmarks.landmark):
                                    if ((landmark.HasField('visibility') and
                                         landmark.visibility < _VISIBILITY_THRESHOLD) or
                                            (landmark.HasField('presence') and
                                             landmark.presence < _PRESENCE_THRESHOLD)):
                                        continue

                                    landmark_pixel = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                                      image_w_cropped, image_h_cropped)

                                    if landmark_pixel is not None:
                                        x_list.append(landmark_pixel[0])
                                        y_list.append(landmark_pixel[1])
                                    else:
                                        continue

                                # Calculate bounding box dimensions
                                bbox_width = max(int(max(x_list)) - int(min(x_list)),
                                                 int(max(y_list)) - int(min(y_list)))
                                bbox_height = bbox_width  # Ensure square bounding box

                                # Calculate bounding box coordinates
                                xmin = max(int(min(x_list)) - 30, 0) + bbox_xmin  # -20
                                xmax = min(int(min(x_list)) + bbox_width + 30, image_w_cropped - 1) + bbox_xmin
                                ymin = max(int(min(y_list)) - 30, 0) + bbox_ymin
                                ymax = min(int(min(y_list)) + bbox_height + 30, image_h_cropped - 1) + bbox_ymin
                                box_arr_hand = np.array([xmin, ymin, xmax, ymax, 0.99])
                                hand_bbox_results.append({'bbox': box_arr_hand})

                        img[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax,:] = img_croppped  # insert img_cropped with hand results to img canvas

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
