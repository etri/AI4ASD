import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import argparse
import json
import timm.models
import numpy as np
import torchvision.transforms as transforms
import math
import gc
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utility import Adder, check_lr
from pathlib import Path
from data.data_load_repdetector import get_dataloaders
from torchvision import models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

from RRBNet.RepDetectModel import RepDetectNet_3D_base_s2

eps = 1e-7


# Parse inputs
def parse_args():
    parser = argparse.ArgumentParser(description='Training pointing pose classifier')
    parser.add_argument('--checkpointdir', type=str, help='output base dir', default='checkpoints')
    parser.add_argument('--j', dest='num_workers', type=int, help='Dataloader CPUS', default=12)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--viz_fps', type=int, default=30)

    # user parameters
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--use_dataparallel', help='Use several GPUs', action='store_true', dest='use_dataparallel',
                        default=False)
    parser.add_argument('--save_freq', type=int, help='save frequency', default=10)
    parser.add_argument('--test_freq', type=int, help='test frequency', default=1)
    parser.add_argument('--batchsize', type=int, help='batch size', default=1)  # 1
    parser.add_argument('--epochs', type=int, help='training epochs', default=20)  # 50
    parser.add_argument('--vid_sample_len', type=int, default=64)
    parser.add_argument('--vid_len_th', type=int, default=900)  # 900
    parser.add_argument('--div_seq_len', type=int, default=900)  # 900
    parser.add_argument('--method', type=int, default=2, choices=[1, 2])
    parser.add_argument('--model_choice', type=str, default='Ours',
                        choices=['Ours'])  # 'P-MUCOS', 'P-MUCOSv2', 'Power spectrum'
    parser.add_argument('--label_shape', type=str, default='matrix', choices=['vector', 'matrix'])

    parser.add_argument('--train_DB', type=list, default=['countix'], choices=['countix', 'QUVA', 'PERTUBE'])
    parser.add_argument('--test_DB', type=list, default=['PERTUBE'], choices=['countix', 'QUVA', 'PERTUBE'])
    parser.add_argument('--root_dir', type=str, help='data choice', default='/home/ych/data/')

    parser.add_argument('--train', action='store_true', default=True, help='training')
    parser.add_argument('--model_name', type=str, default='temp')

    return parser


def update_lr(optimizer, multiplier=.1):
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = param_group['lr'] * multiplier
    optimizer.load_state_dict(state_dict)


def step(model, input_data, optimizer, criteria, is_training):
    # Track history only in training
    with torch.set_grad_enabled(is_training):

        rgb, label = input_data
        output, _ = model(rgb)

        if args.label_shape == 'matrix':
            # reshape per_frame_periodicity to binary matrix
            label_transpose = torch.transpose(label, 0, 1)
            label = torch.matmul(label_transpose, label)
            label = torch.unsqueeze(label, 0)

        else:
            pass
        # Backward
        optimizer.zero_grad()
        loss = criteria(output, label.to(dtype=torch.long))

        del input_data
        del output
        del rgb
        del label
        del _

        # Backward into the branch
        if is_training:
            loss.backward()
            optimizer.step()

    return loss


def _train(model, criteria, optimizer, dataloaders, device=None, num_epochs=200):
    log_dir = args.checkpointdir + '/logs/logs_detector/' + args.model_name
    writer = SummaryWriter(log_dir=log_dir)
    iter_adder = Adder()
    max_iter = len(dataloaders['train'])
    val_acc_best = 0

    for epoch in range(1, num_epochs + 1):

        for phase in ['train']:
            print('Epoch {}, Phase {}'.format(epoch, phase))
            is_training = (phase == 'train')
            model.train(is_training)

            # Learning rate schedule
            # if is_training and (epoch == 20 or epoch == 40):
            if is_training and (epoch == 10):
                update_lr(optimizer, multiplier=.1)

            running_loss = 0.0
            ndata = 0

            # Iterate over data
            for iter_idx, data in enumerate(tqdm(dataloaders[phase])):
                torch.cuda.empty_cache()

                input_data = [data[n].to(device) for n in ['rgb', 'label']]
                loss = step(model, input_data, optimizer, criteria, is_training)

                # Update statistics
                batch_size = input_data[0].size(0)
                running_loss += loss.detach().item() * batch_size
                ndata = ndata + batch_size

                # save loss
                iter_adder(loss.detach().item())
                del input_data
                del data
                del loss

                if (iter_idx + 1) % args.print_freq == 0 and is_training:
                    lr = check_lr(optimizer)
                    print("Epoch: %03d/%03d Iter: %4d/%4d  Loss: %7.4f  LR:%.10f" % (
                        epoch, args.epochs, iter_idx + 1, max_iter,
                        iter_adder.average(), lr))

                    writer.add_scalar('Loss_{}/iter'.format('train'), iter_adder.average(),
                                      iter_idx + (epoch - 1) * max_iter)

                    iter_adder.reset()

            epoch_loss = running_loss / ndata
            print('Loss_epoch: {}'.format(epoch_loss))

            # tensorboard update
            writer.add_scalar('Loss_{}/epoch'.format(phase), epoch_loss, epoch)
            writer.flush()

        if epoch == 1 or epoch % args.test_freq == 0:


            print("validating model...")
            val_acc = _test_model(model, dataloaders, args, device, 'val')

            print('val_Acc: {:.3f}'.format(val_acc[0]))
            print('recall: {:.3f}'.format(val_acc[1]))
            print('precision: {:.3f}'.format(val_acc[2]))
            print('fscore: {:.3f}'.format(val_acc[3]))
            print('overlap: {:.3f}'.format(val_acc[4]))

            fp = open(os.path.join('checkpoints/logs/logs_detector', args.model_name, 'Results_val.txt'), 'a')
            fp.write('Epoch:%d ' % (epoch))
            fp.write('val_Accuracy: {:.3f}, recall: {:.3f}, precision: {:.3f}, fscore: {:.3f}, overlap: {:.3f} \n'
                     .format(val_acc[0], val_acc[1], val_acc[2], val_acc[3], val_acc[4]))
            fp.close()
            writer.add_scalar('Acc_val/epoch', val_acc[0], epoch)

            ## select best model on testidation sets ## no test set -> just final epoch
            if val_acc[0] > val_acc_best:
                val_acc_best = val_acc[0]
                filename = (args.checkpointdir + '/logs/logs_detector/' + args.model_name + '/model_best.checkpoint')
                torch.save(model.state_dict(), filename)
                print('Saving ' + filename)

                # filename = (args.checkpointdir + '/logs/logs_detector/' + args.model_name + '/model_%d.checkpoint' % epoch)
                # torch.save(model.state_dict(), filename)
                # print('Saving ' + filename)

                print('testing with current best test model')
                test_acc = _test_model(model, dataloaders, args, device, 'test')

                print('test_Acc: {:.3f}'.format(test_acc[0]))
                print('recall: {:.3f}'.format(test_acc[1]))
                print('precision: {:.3f}'.format(test_acc[2]))
                print('fscore: {:.3f}'.format(test_acc[3]))
                print('overlap: {:.3f}'.format(test_acc[4]))

                fp = open(os.path.join('checkpoints/logs/logs_detector', args.model_name, 'Results_test.txt'), 'a')
                fp.write('Epoch:%d ' % (epoch))
                fp.write('test_Accuracy: {:.3f}, recall: {:.3f}, precision: {:.3f}, fscore: {:.3f}, overlap: {:.3f} \n'
                         .format(test_acc[0], test_acc[1], test_acc[2], test_acc[3], test_acc[4]))
                fp.close()


def _test(model, dataloaders, phase, device=None):
    acc = 0
    precision = 0
    recall = 0
    fscore = 0
    overlap = 0
    ndata = 0
    vid_len_th = args.vid_len_th

    model.train(True)

    with torch.no_grad():
        for iter_idx, data in enumerate(tqdm(dataloaders[phase])):
            torch.cuda.empty_cache()
            rgb, label = [data[n].to(device) for n in ['rgb', 'label']]

            video_len = rgb.shape[1]
            ndata += rgb.size(0)

            if args.model_choice == 'Ours':

                if video_len <= vid_len_th:
                    output, _ = model(rgb)  # 56, 56, 300
                    preds = torch.argmax(output, 1)

                    del output
                    del _

                else:

                    strides = [1]

                    preds = torch.zeros(1, video_len, video_len).cuda()
                    preds_squeeze = torch.zeros(video_len)

                    for j in range(len(strides)):
                        print("current stride:{} ".format(strides[j]))

                        loop_num = int(video_len / args.div_seq_len / strides[j] + 0.5)
                        seg_len = args.div_seq_len * strides[j]

                        for i in range(loop_num):
                            print("current loop:{} / total loop:{}".format(i + 1, loop_num))

                            # Forward
                            if i == loop_num - 1:
                                rgb_sub = rgb[:, (i) * seg_len::strides[j], :, :, :]
                            else:
                                rgb_sub = rgb[:, (i) * seg_len:(i + 1) * seg_len:strides[j], :, :, :]

                            output_sub, _ = model(rgb_sub)
                            preds_sub = torch.argmax(output_sub, 1)

                            H = preds_sub.shape[1]
                            W = preds_sub.shape[2]

                            if strides[j] != 1:
                                resize = transforms.Resize((H * strides[j], W * strides[j]), interpolation=transforms.InterpolationMode.NEAREST)
                                preds_sub = resize(preds_sub)

                            if i == loop_num - 1:

                                sub_len = video_len - (i) * seg_len
                                preds[:, (i) * seg_len:, (i) * seg_len:] += preds_sub[:, :sub_len, :sub_len]
                            else:
                                preds[:, (i) * seg_len:(i + 1) * seg_len, (i) * seg_len:(i + 1) * seg_len] += preds_sub

                            del output_sub
                            del preds_sub
                            del rgb_sub
                            del _
                    # if 'inference' in args.stride_sampling:
                    #     preds = (preds >= 2.0).float()

                del rgb
                del data

                # thresholding ## ??
                preds = (preds >= 0.5).float()

                # Method1
                if args.method == 1:
                    preds_squeeze, _ = torch.max(preds, dim=-1)
                    preds_squeeze = torch.squeezeload_video(preds_squeeze, dim=0)

                # Method2
                elif args.method == 2:

                    preds_squeeze = torch.squeeze(preds, dim=0)
                    preds_squeeze = torch.diagonal(preds_squeeze, 0)

                preds_squeeze = preds_squeeze.cpu().numpy()

            label = label.detach().cpu().numpy()

            # calculate accuracy before smoothing
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for i in range(preds_squeeze.size):

                if preds_squeeze[i] == label[0, i]:
                    if preds_squeeze[i] == 1:
                        TP = TP + 1
                    else:
                        TN = TN + 1
                else:
                    if preds_squeeze[i] == 1:
                        FP = FP + 1
                    else:
                        FN = FN + 1

            del preds_squeeze
            del preds
            del label

            precision = precision + TP / (TP + FP + eps)
            recall = recall + TP / (TP + FN + eps)
            precision_now = TP / (TP + FP + eps)
            recall_now = TP / (TP + FN + eps)

            acc = acc + (TP + TN) / (TP + FN + FP + TN + eps)
            fscore = fscore + 2 * (precision_now * recall_now) / (precision_now + recall_now + eps)
            overlap = overlap + TP / (TP + FN + FP + eps)

        precision = precision / ndata
        recall = recall / ndata
        acc = acc / ndata
        fscore = fscore / ndata
        overlap = overlap / ndata

        acc_list = [acc, recall, precision, fscore, overlap]

    return acc_list


def _test_model(model, dataloaders, args, device, phase):
    # if args.train is False:
    if phase == 'test':  # 'test'

        testmodel_filename = (args.checkpointdir + '/logs/logs_detector/' + args.model_name + '/model_best.checkpoint')
        #testmodel_filename = (args.checkpointdir + '/logs/logs_detector/' + args.model_name + '/model_20.checkpoint')


        print('Loading from: ', testmodel_filename)
        state_dict = torch.load(testmodel_filename)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        state_dict = {k: v for k, v in state_dict.items() if
                      (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        model.cuda()
    else:
        pass

    test_model_acc = _test(model, dataloaders, phase, device=device)

    gc.collect()
    torch.cuda.empty_cache()

    return test_model_acc


def _train_model(model, dataloaders, args, device):
    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # hardware tuning
    if torch.cuda.device_count() > 1 and args.use_dataparallel:
        model = torch.nn.DataParallel(model)

    model.cuda()

    _train(model, criteria, optimizer, dataloaders, device=device, num_epochs=args.epochs)


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    print("The configuration of this run is:")
    print(args, end='\n\n')

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")  # cuda:1
    dataloaders = get_dataloaders(args)

    model = RepDetectNet_3D_base_s2(args)

    if args.train:

        train_path = os.path.join('checkpoints/logs/logs_detector', args.model_name)
        Path(train_path).mkdir(parents=True, exist_ok=True)
        train_args_dir = os.path.join(train_path, 'train_args.txt')

        with open(train_args_dir, 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        print("Saving train args")
        print("Training Proposed network")
        print("Train model: {}".format(args.model_choice))
        print("TrainDB: {}".format(args.train_DB))
        print("TestDB: {}".format(args.test_DB))
        print("Model Name: {}".format(args.model_name))

        _train_model(model, dataloaders, args, device)
    else:

        with open(os.path.join('checkpoints/logs/logs_detector', args.model_name, 'test_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        print("Saving test args")
        print("Evaluating single model...")
        print("TestDB: {}".format(args.test_DB))
        print("Model Name: {}".format(args.model_name))

        test_acc = _test_model(model, dataloaders, args, device, 'test')  # 'test
        print('test_Acc: {:.3f}'.format(test_acc[0]))
        print('recall: {:.3f}'.format(test_acc[1]))
        print('precision: {:.3f}'.format(test_acc[2]))
        print('fscore: {:.3f}'.format(test_acc[3]))
        print('overlap: {:.3f}'.format(test_acc[4]))

        fp = open(os.path.join('checkpoints/logs/logs_detector', args.model_name, 'Results_test.txt'), 'a')
        fp.write('Epoch:best ')
        fp.write('test_Accuracy: {:.3f}, recall: {:.3f}, precision: {:.3f}, fscore: {:.3f}, overlap: {:.3f} \n'
                 .format(test_acc[0], test_acc[1], test_acc[2], test_acc[3], test_acc[4]))
        fp.close()