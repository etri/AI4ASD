import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch.backends.cudnn as cudnn
import argparse
import torch.nn.functional as F
import warnings
import gc
from RRBNet.RepDetectModel import  RepDetectNet_3D_base_s2 #

from RRBNet.RepClsModel import VideoSwinTransformerModel
from tqdm import tqdm
from utility import *
from data.data_load_demo import get_dataloaders, NormalizeLen, video_transform
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score


warnings.filterwarnings(action='ignore')
m = nn.Softmax(dim=1)
eps = 1e-7


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='Batch Size.', type=int, default=1)
    parser.add_argument('--j', dest='num_workers', type=int, help='Dataloader CPUS', default=0)

    # User parameters (thresholds)
    parser.add_argument('--data_choice', type=str, help='data choice', default='SSBD', choices=['SSBD', 'ESBD'])
    parser.add_argument('--input', type=str, default='video', help='input_data_select', choices=['mkv', 'video'])

    # Parameters (RepDetectNet + RepNet)
    parser.add_argument('--max_frame_len', type=int, default=54000000) ####### for long long video, 9000(5minute), 5400(3minute)
    parser.add_argument('--vid_len_th', type=int, default=900)  # 900
    parser.add_argument('--div_seq_len', type=int, default=900)  # 900
    parser.add_argument('--normalize_vid_len', type=int, default=16)  #videomae:16, VST/I3D: 64
    parser.add_argument('--consecutive_len_th', type=int, default=30)  # 30
    parser.add_argument('--median_filter_w_size', type=int, default=51)  ########### 9, 31, 51
    parser.add_argument('--method', type=int, default=2, choices=[1, 2])
    parser.add_argument('--aggre_method', type=str, default='all', choices=['top-1', 'top-k', 'all'])
    parser.add_argument('--detect_periodicity', type=str, default='detect', choices=['detect', 'uniform', 'gt'])
    parser.add_argument('--proposal_conf_th', type=float, default=0.0) #0.79:real-world, 0:public
    parser.add_argument('--prob_conf_th', type=float, default=0.0) #0.5:real-world, 0:public

    # Visualize Flag
    parser.add_argument('--showFps', type=bool, default=False)
    parser.add_argument('--viz_scale', type=int, default=1)  # 2
    parser.add_argument('--viz_skeleton', type=bool, default=False)
    parser.add_argument('--visualize_red_dot', type=bool, default=False)

    parser.add_argument('--visualize_result', type=bool, default=True)
    parser.add_argument('--saveVideo', type=bool, default=False)
    parser.add_argument('--saveResult', type=bool, default=False)
    parser.add_argument('--visualize_screenshot', type=bool, default=False)

    return parser


if __name__ == '__main__':

    parser = parse_args()
    args = parser.parse_args()
    args.datadir = os.path.join('/home/ych/data', args.data_choice)
    args.annotate_dir = args.datadir + '/Annotations_revised_ych/'

    print("The configuration of this run is:")
    print(args, end='\n\n')

    torch.set_grad_enabled(False)
    cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu") # cuda:1

    model_face_touch = None
    preprocess = None

    model_RepDetector = RepDetectNet_3D_base_s2(args)
    state_dict = torch.load('checkpoints/RRB_LA_Net_tr_countix.checkpoint')
    model_RepDetector.load_state_dict(state_dict)
    model_RepDetector.cuda()
    model_RepDetector.train(True)
    # model_RepDetector.eval()
    print('Loading RepDetector model done...')

    # 반복적 행동 인식 모델

    if args.data_choice == 'SSBD':# or args.data_choice == 'DEMO':
        ## 3-class 분류기 trained on ESBD(arm flapping(+hand action), spinning, headbanging)
        model_RePNet = VideoSwinTransformerModel(mode='demo')
        action_cls_dict = {'ArmFlapping': 0, 'Spinning': 1, 'HeadBanging': 2}
        action_cls = ['ArmFlapping', 'Spinning', 'HeadBanging']
        #action_cls = ['ArmFlapping', 'Spinning', 'HeadBanging', 'ToyPlaying', 'Jumping', 'etc']
        state_dict = torch.load('checkpoints/RRB_RA_Net_tr_ESBD_parsing.checkpoint')

    elif args.data_choice == 'ESBD':
        ## 3-class 분류기 trained on SSBD(arm flapping, spinning, headbanging)
        model_RePNet = VideoSwinTransformerModel(mode='demo')
        action_cls_dict = {'ArmFlapping': 0, 'Spinning': 1, 'HeadBanging': 2}
        action_cls = ['ArmFlapping', 'Spinning', 'HeadBanging']
        #action_cls = ['ArmFlapping', 'Spinning', 'HeadBanging', 'ToyPlaying', 'Jumping', 'etc']
        state_dict = torch.load('checkpoints/RRB_RA_Net_tr_SSBD_parsing.checkpoint)


    model_RePNet.load_state_dict(state_dict)
    model_RePNet.cuda()
    model_RePNet.eval()
    print('Loading RepNet model done...')

    dataloaders = get_dataloaders(args)

    ndata = 0
    running_corrects = 0
    acc_pp = 0
    precision_pp = 0
    recall_pp = 0
    fscore_pp = 0
    overlap = 0

    vid_level_preds = []
    vid_level_labels = []

    video_lst_pred, start_lst_pred, end_lst_pred, label_lst_pred, score_lst_pred = [], [], [], [], []
    video_lst_gt, start_lst_gt, end_lst_gt, label_lst_gt = [], [], [], []

    with torch.no_grad():
        #with autocast():
        for iter_idx, data in enumerate(tqdm(dataloaders['test'])):

            gc.collect()
            torch.cuda.empty_cache()

            rgb, gt, label = [data[n] for n in ['rgb', 'gt', 'label']]  # gt: per_frame_periodicity, label: action class

            rgb = np.expand_dims(rgb, axis=0)
            rgb_path = data['rgbpath']
            path_tok = rgb_path.split('/')
            path_tok_file = path_tok[-1].split('.')
            print('analyzing video: {}'.format(rgb_path))

            video_len = rgb.shape[1]
            vid_len_th = args.vid_len_th
            proposal_prob = torch.zeros(video_len)

            if video_len <= vid_len_th and video_len >= 30:

                rgb_tensor = torch.tensor(np.array(rgb)).float().cuda()
                output, ssm = model_RepDetector(rgb_tensor)  # 56, 56, 300
                _, preds = torch.max(output, 1) #############

                ## max value
                ## Method1
                if args.method == 1:
                    preds_squeeze, _ = torch.max(preds, dim=-1)
                    preds_squeeze = torch.squeeze(preds_squeeze, dim=0)

                ## Method2
                elif args.method == 2:
                    preds_squeeze = torch.squeeze(preds, dim=0)
                    preds_squeeze = torch.diagonal(preds_squeeze, 0)

                out_softmax = F.softmax(output, dim=1)
                for i in range(video_len):
                    proposal_prob[i] = out_softmax[:, 1, i, i]

                del output
                #del preds
                #del ssm
                gc.collect()
                torch.cuda.empty_cache()

            else:
                seg_len = args.div_seq_len
                loop_num = int(video_len / seg_len + 0.5)
                ssm = torch.zeros(1, 1, video_len, video_len)
                preds = torch.zeros(1, video_len, video_len)
                preds_squeeze = torch.zeros(video_len)

                for i in range(loop_num):
                    print("current loop:{} / total loop:{}".format(i + 1, loop_num))

                    # Forward
                    if i == loop_num - 1:
                        rgb_sub = rgb[:, (i) * seg_len:, :, :, :]
                    else:
                        rgb_sub = rgb[:, (i) * seg_len:(i + 1) * seg_len, :, :, :]

                    rgb_sub = torch.tensor(np.array(rgb_sub)).float().cuda()

                    output_sub, ssm_sub = model_RepDetector(rgb_sub)
                    _, preds_sub = torch.max(output_sub, 1) ###################

                    if i == loop_num - 1:
                        preds[:, (i) * seg_len:, (i) * seg_len:] = preds_sub
                        ssm[:, :, (i) * seg_len:, (i) * seg_len:] = ssm_sub
                    else:
                        preds[:, (i) * seg_len:(i + 1) * seg_len, (i) * seg_len:(i + 1) * seg_len] = preds_sub
                        ssm[:, :, (i) * seg_len:(i + 1) * seg_len, (i) * seg_len:(i + 1) * seg_len] = ssm_sub

                    # Method1
                    if args.method == 1:
                        preds_squeeze_sub, _ = torch.max(preds_sub, dim=-1)
                        preds_squeeze_sub = torch.squeeze(preds_squeeze_sub, dim=0)

                    # Method2
                    elif args.method == 2:

                        preds_squeeze_sub = torch.squeeze(preds_sub, dim=0)
                        preds_squeeze_sub = torch.diagonal(preds_squeeze_sub, 0)

                    proposal_prob_sub = torch.zeros(preds_squeeze_sub.shape[0])

                    if i == loop_num - 1:
                        preds_squeeze[(i) * seg_len:] = preds_squeeze_sub

                        out_sub_softmax = F.softmax(output_sub, dim=1)
                        for j in range(preds_squeeze_sub.shape[0]):
                            proposal_prob_sub[j] = out_sub_softmax[:, 1, j, j]
                        proposal_prob[(i) * seg_len:] = proposal_prob_sub

                    else:
                        preds_squeeze[(i) * seg_len:(i + 1) * seg_len] = preds_squeeze_sub

                        out_sub_softmax = F.softmax(output_sub, dim=1)
                        for j in range(preds_squeeze_sub.shape[0]):
                            proposal_prob_sub[j] = out_sub_softmax[:, 1, j, j]
                        proposal_prob[(i) * seg_len:(i + 1) * seg_len] = proposal_prob_sub

                    del output_sub
                    del preds_sub
                    del rgb_sub
                    del ssm_sub
                    gc.collect()
                    torch.cuda.empty_cache()

            # del model_RepDetector
            gc.collect()
            torch.cuda.empty_cache()

            preds_squeeze = preds_squeeze.cpu().numpy()
            gt_numpy = gt
            ndata += rgb.shape[0]

            # calculate accuracy before smoothing
            TP = 0; TN = 0; FP = 0; FN = 0

            gt_numpy_cast = np.zeros_like((gt_numpy))
            for i in range((gt_numpy.shape[0])):
                if gt_numpy[i] == 0:
                    gt_numpy_cast[i] = 0
                else:
                    gt_numpy_cast[i] = 1

            for i in range(preds_squeeze.size):

                if preds_squeeze[i] == gt_numpy_cast[i]:
                    if preds_squeeze[i] == 1:
                        TP = TP + 1
                    else:
                        TN = TN + 1
                else:
                    if preds_squeeze[i] == 1:
                        FP = FP + 1
                    else:
                        FN = FN + 1

            precision_pp = precision_pp + TP / (TP + FP + eps)
            recall_pp = recall_pp + TP / (TP + FN + eps)
            precision_now = TP / (TP + FP + eps)
            recall_now = TP / (TP + FN + eps)

            acc_pp = acc_pp + (TP + TN) / (TP + FN + FP + TN + eps)
            fscore_pp = fscore_pp + 2 * (precision_now * recall_now) / (precision_now + recall_now + eps)
            overlap = overlap + TP / (TP + FN + FP + eps)

            # precision, recall, fscore, _ = score(gt_numpy[0], preds_squeeze, average=None)
            # confusion_mat = confusion_matrix(gt_numpy[0], preds_squeeze)

            ### smoothing results
            preds_squeeze = preds_squeeze.astype(np.uint8)
            preds_squeeze = cv2.medianBlur(preds_squeeze, args.median_filter_w_size)  #9

            if args.detect_periodicity == 'detect':

                positions = defaultdict(set)
                for index, value in enumerate(preds_squeeze[:,0]):
                    positions[value].add(index)

                periodic_segment_list = ranges(positions[1])

            elif args.detect_periodicity == 'uniform':

                periodic_segment_list = []
                segment_num = math.ceil(video_len / vid_len_th)
                for i in range(segment_num):
                    if i == segment_num - 1:
                        periodic_segment_list.append((i * vid_len_th, video_len))
                    else:
                        periodic_segment_list.append((i * vid_len_th, (i + 1) * vid_len_th))

            elif args.detect_periodicity == 'gt':
                positions = defaultdict(set)
                for index, value in enumerate(gt_numpy[0,:]):
                    positions[value].add(index)

                periodic_segment_list = ranges(positions[1])

            consecutive_len_max = 0

            for i in range(len(periodic_segment_list)):  ## 비디오에서 제안된 구간 중 max 길이인것 get
                consecutive_len = periodic_segment_list[i][1] - periodic_segment_list[i][0]
                if consecutive_len >= consecutive_len_max:
                    consecutive_len_max = consecutive_len

            consecutive_len_th = args.consecutive_len_th
            if consecutive_len_max < consecutive_len_th:
                consecutive_len_th = consecutive_len_max

            output_softmax_list = []
            proposal_softmax_list = []
            result_list = []

            ## loop for action classification in each proposed segment
            for i in range(len(periodic_segment_list)):

                start_idx = periodic_segment_list[i][0]
                end_idx = periodic_segment_list[i][1]

                consecutive_len = end_idx - start_idx
                if consecutive_len >= consecutive_len_th: # 필요한가?
                    pass
                else:
                    continue

                ## RepNet ##
                rgb_sub = rgb[:, start_idx:end_idx, :, :, :]
                rgb_sub = NormalizeLen(rgb_sub, vid_len=args.normalize_vid_len)  # 224, 224, 64

                output_rgb = model_RePNet(rgb_sub)
                action_preds = torch.argmax(output_rgb, dim=-1)
                action_preds = action_preds.cpu().detach().numpy()

                ## segment probability
                proposal_prob_ave = np.mean(proposal_prob[start_idx:end_idx].numpy())

                output_softmax = softmax(output_rgb.cpu().numpy())
                output_softmax_cls_preds = output_softmax[0][action_preds]
                output_softmax_list.append(output_softmax)
                proposal_softmax_list.append(proposal_prob_ave)

                del output_rgb
                del rgb_sub
                gc.collect()
                torch.cuda.empty_cache()

                result_list.append({'start_idx': start_idx, 'end_idx': end_idx, 'action_preds': action_preds, 'prob': output_softmax_cls_preds, 'proposal_conf': proposal_prob_ave})

                #### crawling prediction data for calculating mAP
                video_lst_pred.append(path_tok_file[0])
                start_lst_pred.append(start_idx)
                end_lst_pred.append(end_idx)
                label_lst_pred.append(action_preds[0])
                score_lst_pred.append(1.0)

            ## sorting
            sorted_idx = np.argsort(proposal_softmax_list)[::-1]

            if args.aggre_method == 'top-1':
                top_k = 1
            elif args.aggre_method == 'top-k':
                top_k = min(3, len(sorted_idx))
            elif args.aggre_method == 'all':
                top_k = len(sorted_idx)

            prob_aggre = 0
            for i in range(top_k):
                prob_aggre += output_softmax_list[sorted_idx[i]]
            prob_aggre = prob_aggre / top_k

            vid_level_pred = np.argmax(prob_aggre)
            print('video_level_cls_pred:', action_cls[vid_level_pred])

            if int(label) == -1:
                action_cls_label = -1
                print('Video_level_cls_label:', action_cls_label)
            else:
                action_cls_label = action_cls[label]
                print('video_level_cls_label:', action_cls_label)

            correct = (vid_level_pred == label)
            prob = int(prob_aggre[0][vid_level_pred] * 1000) / 10.0

            fp = open(os.path.join('checkpoints/logs/logs_demo', args.data_choice + '_results_test.txt'), 'a')
            fp.write('path: {}, pred: {}, g.t.: {}, correct: {}, prob(%): {} \n'.format(rgb_path,
                                                                                        action_cls[vid_level_pred],
                                                                                        action_cls_label,
                                                                                        correct, prob))
            fp.close()

            vid_level_preds.append(vid_level_pred)
            vid_level_labels.append(label)
            #running_corrects += np.sum(vid_level_pred == label[0])
            running_corrects += np.sum(vid_level_pred == label)

            gc.collect()
            torch.cuda.empty_cache()

            if args.visualize_result == True:
                visualize_result(rgb_path, gt, args, result_list)

        ## Final accuracy on video level action classification(ac)
        precision_ac, recall_ac, fscore_ac, _ = score(vid_level_labels, vid_level_preds, average="weighted")

        acc_ac = running_corrects / float(ndata)
        confusion_mat = confusion_matrix(vid_level_labels, vid_level_preds)
        print('------Final accuracy on video level action classification------')
        print('Acc_ac: {:.3f}'.format(acc_ac))
        print('precision_ac: {:.3f}'.format(precision_ac))
        print('recall_ac: {:.3f}'.format(recall_ac))
        print('fscore_ac: {:.3f}\n'.format(fscore_ac))

        print('------Confusion matrix------')
        print(confusion_mat)
        print(classification_report(vid_level_labels, vid_level_preds, target_names=action_cls))#['ArmFlapping', 'Spinning', 'HeadBanging']))

        # save confusion matrix
        np_txt = os.path.join('checkpoints/logs/logs_demo', args.data_choice + '_confusion.txt')
        np.savetxt(np_txt, confusion_mat)

        ## Final accuracy on per_frame_periodicity_predction(pp)
        precision_pp = precision_pp / ndata
        recall_pp = recall_pp / ndata
        acc_pp = acc_pp / ndata
        fscore_pp = fscore_pp / ndata
        overlap = overlap / ndata

        print('------Final accuracy on per_frame_periodicity_prediction------')
        print('Acc_pp: {:.3f}'.format(acc_pp))
        print('precision_pp: {:.3f}'.format(precision_pp))
        print('recall_pp: {:.3f}'.format(recall_pp))
        print('fscore_pp: {:.3f}'.format(fscore_pp))
        print('overlap: {:.3f}\n'.format(overlap))

        fp = open(os.path.join('checkpoints/logs/logs_demo', args.data_choice + '_results_test.txt'), 'a')
        fp.write('Final accuracy on mAP@tIOU \n')
        fp.write('Final accuracy on video level action classification \n')
        fp.write('test_Accuracy: {:.3f}, recall: {:.3f}, precision: {:.3f}, f_score: {:.3f} \n'.format(acc_ac, recall_ac, precision_ac, fscore_ac))
        fp.write('Final accuracy on per_frame_periodicity_prediction \n')
        fp.write('test_Accuracy: {:.3f}, recall: {:.3f}, precision: {:.3f}, f_score: {:.3f}, overlap: {:.3f} \n'.format(acc_pp, recall_pp, precision_pp, fscore_pp, overlap))
        fp.close()

