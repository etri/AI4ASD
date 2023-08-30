"""
   * Source: main.py
   * License: PBR4AI License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 21 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import argparse
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utility import Adder, check_lr, loss_negative_cosine
from pathlib import Path
from data.data_load import get_dataloaders
from model.model import build_net

eps = 1e-7

# %% Parse inputs
def parse_args():

    """ parse_args function

    Note: function for user parameter setting

    """

    parser = argparse.ArgumentParser(description='Training pointing pose classifier')
    parser.add_argument('--checkpointdir', type=str, help='output base dir', default='checkpoints')
    parser.add_argument('--j', dest='num_workers', type=int, help='Dataloader CPUS', default=12)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--viz_fps', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--use_dataparallel', help='Use several GPUs', action='store_true', dest='use_dataparallel',
                        default=False)

    parser.add_argument('--save_freq', type=int, help='save frequency', default=10)
    parser.add_argument('--test_freq', type=int, help='test frequency', default=1)
    parser.add_argument('--batchsize', type=int, help='batch size', default=4) #8
    parser.add_argument('--epochs', type=int, help='training epochs', default=2)
    parser.add_argument('--SSL', type=str, default='SimSiam', choices=['None', 'SimSiam', 'BYOL'])
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'vit_B_32'])

    # model name
    parser.add_argument('--model_name', type=str, default='resnet50_ntu_SimSiam')

    return parser


def update_lr(optimizer, multiplier=.1):

    """update_lr function

    Note: function for update_lr

    """

    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = param_group['lr'] * multiplier
    optimizer.load_state_dict(state_dict)


def step(model, input_data, optimizer, criteria, is_training):

    """step function

    Note: function for calculating loss and backward

    """

    # Track history only in training
    with torch.set_grad_enabled(is_training):

        if args.SSL == 'None':
            rgb, label = input_data
            output = model(rgb)
            loss_metric = 0
        else:
            rgb, label, rgb_aux = input_data
            output, metric_feature = model(rgb, rgb_aux)

            if args.SSL == 'BYOL':
                loss_metric = metric_feature
            elif args.SSL == 'SimSiam':
                z1 = metric_feature[0]
                z2 = metric_feature[1]
                p1 = metric_feature[2]
                p2 = metric_feature[3]
                loss_metric = (loss_negative_cosine(p1, z2, label) + loss_negative_cosine(p2, z1, label)) / 2.0

        _, preds = torch.max(output, 1)

        # Backward
        optimizer.zero_grad()
        loss = criteria(output, label)

        weight_factor = 0.5

        # aux loss
        loss_total = loss + weight_factor * loss_metric

        # Backward into the branch
        if is_training:
            loss_total.backward()
            optimizer.step()

            ## BYOL
            if args.SSL == 'BYOL':
                model.update_moving_average()

    return loss, preds


def train(model, criteria, optimizer, dataloaders, device=None, num_epochs=200):

    """train function

    Note: main function for training the pointing recognition network

    """

    best_loss = float('inf')
    # setup tensorboard
    log_dir = args.checkpointdir + '/logs/' + args.model_name
    writer = SummaryWriter(log_dir=log_dir)

    iter_adder_visual = Adder()
    max_iter = len(dataloaders['train'])
    test_acc_best = 0

    for epoch in range(1, num_epochs + 1):
        for phase in ['train']:
            print('Epoch {}, Phase {}'.format(epoch, phase))
            is_training = (phase == 'train')
            model.train(is_training)

            # Learning rate schedule
            if is_training and (epoch == 20 or epoch == 40):
                update_lr(optimizer, multiplier=.1)

            running_loss = 0.0
            running_corrects = 0
            ndata = 0

            # Iterate over data
            for iter_idx, data in enumerate(tqdm(dataloaders[phase])):

                if args.SSL == 'None':
                    input_data = [data[n].to(device) for n in ['rgb', 'label']]
                else:
                    input_data = [data[n].to(device) for n in ['rgb', 'label', 'rgb_aux']]

                # # Update Visual Branch
                loss, preds = step(model, input_data, optimizer, criteria, is_training)

                # Update statistics
                batch_size = input_data[0].size(0)
                running_loss += loss.item() * batch_size

                if args.SSL == "None":
                    running_corrects += torch.sum(preds == input_data[-1].data)
                else:
                    running_corrects += torch.sum(preds == input_data[-2].data)

                ndata = ndata + batch_size

                # save loss
                iter_adder_visual(loss.item())

                if (iter_idx + 1) % args.print_freq == 0 and is_training:
                    lr = check_lr(optimizer)
                    print(
                        "Epoch: %03d/%03d Iter: %4d/%4d  Loss: %7.4f  LR:%.10f" % (
                            epoch, args.epochs, iter_idx + 1, max_iter,
                            iter_adder_visual.average(),  lr))

                    writer.add_scalar('Loss_{}/iter'.format('train'), iter_adder_visual.average(),
                                      iter_idx + (epoch - 1) * max_iter)

                    iter_adder_visual.reset()

            epoch_loss = running_loss / ndata
            epoch_acc = running_corrects.double() / ndata

            print('Loss: {}, Accuracy: {}'.format(epoch_loss, epoch_acc))

            # tensorboard update
            writer.add_scalar('Acc_{}/epoch'.format(phase), epoch_acc, epoch)
            writer.add_scalar('Loss_{}/epoch'.format(phase), epoch_loss, epoch)
            writer.flush()

        if epoch == 1 or epoch % args.test_freq == 0:
            # validate
            print("Validaing single model...")
            test_acc = test_model(model, dataloaders, device, 'val')
            fp = open(os.path.join('checkpoints/logs', args.model_name, 'Results_val.txt'), 'a')
            fp.write('Epoch:%d ' % (epoch))
            fp.write(
                'TP: {}, FP: {}, TN: {}, FN: {} '.
                    format(test_acc[4], test_acc[5], test_acc[6], test_acc[7]))
            fp.write('val_Accuracy: {:.3f}, recall: {:.3f}, precision: {:.3f}, f_score: {:.3f} \n'.format(
                    test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
            fp.close()

            print('val_Accuracy: {:.3f}, recall: {:.3f}, precision: {:.3f}, f_score: {:.3f}'.format(
                test_acc[0], test_acc[1], test_acc[2], test_acc[3]))

            writer.add_scalar('Acc_val/epoch', test_acc[0], epoch)

            ## select best model on validation sets
            if test_acc[0] > test_acc_best:

                test_acc_best = test_acc[0]

                filename = (args.checkpointdir + '/logs/' + args.model_name + '/model_best.checkpoint')
                torch.save(model.state_dict(), filename)
                print('Saving ' + filename)

            ## select model on final epoch
            if epoch == num_epochs:

                filename = (args.checkpointdir + '/logs/' + args.model_name + '/model_%d.checkpoint' % epoch)
                torch.save(model.state_dict(), filename)
                print('Saving ' + filename)

    return best_loss


def test(model, dataloaders, phase, device=None):

    """train function

    Note: function for testing(validating) the pointing recognition network

    """

    model.train(False)
    running_corrects = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    i=0

    # Iterate over data
    ndata = 0
    with torch.no_grad():
        for iter_idx, data in enumerate(tqdm(dataloaders[phase])):

            if args.SSL == 'None':
                rgb, label = [data[n].to(device) for n in ['rgb', 'label']]
                output = model(rgb)

            else:
                rgb, label, rgb_aux = [data[n].to(device) for n in ['rgb', 'label', 'rgb_aux']]
                output, _ = model(rgb, rgb_aux)

            preds = torch.argmax(output, dim=-1)

            # Update statistics
            running_corrects += torch.sum(preds == label.data)
            ndata += rgb.size(0)

            if preds == label.data:  # binary classification
                if preds == 0:
                    TP = TP + 1
                else:
                    TN = TN + 1
            else:
                if preds == 0:
                    FP = FP + 1
                else:
                    FN = FN + 1

            i += 1

    acc = running_corrects.double() / ndata
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f_score = 2*(precision * recall) / (precision + recall + eps)
    acc_list = [acc, recall, precision, f_score, TP, FP, TN, FN]
    return acc_list


def test_model(model, dataloaders, device, phase):

    """test_model function

    Note: function for test_model

    """

    test_model_acc = test(model, dataloaders, phase, device=device)

    return test_model_acc


def train_model(model, dataloaders, args, device):

    """train_model function

    Note: function for train_model

    """

    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # hardware tuning
    if torch.cuda.device_count() > 1 and args.use_dataparallel:
        model = torch.nn.DataParallel(model)

    model.cuda()

    val_model_acc = train(model, criteria, optimizer, dataloaders, device=device, num_epochs=args.epochs)
    return val_model_acc


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    print("The configuration of this run is:")
    print(args, end='\n\n')

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu") # cuda:1

    model = build_net(args)

    ## Params (MB)
    num_of_parameters = sum(map(torch.numel, model.parameters()))
    print("Total paramter(MB):{}".format(num_of_parameters * 4 / 1024 / 1024.0))

    dataloaders = get_dataloaders(args)

    train_path = os.path.join('checkpoints/logs', args.model_name)
    Path(train_path).mkdir(parents=True, exist_ok=True)

    train_args_dir = os.path.join(train_path, 'train_args.txt')

    with open(train_args_dir, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print("Saving train args")
    print("Training Proposed network")

    train_model(model, dataloaders, args, device)

