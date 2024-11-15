import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import argparse
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utility import Adder, check_lr
from pathlib import Path
from data.data_load_classifier import get_dataloaders
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from RRBNet.RepClsModel import VideoSwinTransformerModel, VideoMAE, I3D
from thop import profile
from ptflops import get_model_complexity_info

eps = 1e-7

# Parse inputs
def parse_args():

    parser = argparse.ArgumentParser(description='Training pointing pose classifier')
    parser.add_argument('--checkpointdir', type=str, help='output base dir', default='checkpoints')
    parser.add_argument('--j', dest='num_workers', type=int, help='Dataloader CPUS', default=12) #12
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--viz_fps', type=int, default=30)

    # user parameters
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--use_dataparallel', help='Use several GPUs', action='store_true', dest='use_dataparallel', default=False)
    parser.add_argument('--save_freq', type=int, help='save frequency', default=10)
    parser.add_argument('--test_freq', type=int, help='test frequency', default=1)
    parser.add_argument('--batchsize', type=int, help='batch size', default=1)#8
    parser.add_argument('--epochs', type=int, help='training epochs', default=50)
    parser.add_argument('--vid_sample_len', type=int, default=64) #videomae: 16,  VST/I3D: 64,
    parser.add_argument('--backbone', type=str, default='VST', choices=['VST', 'VideoMAE',  'I3D'])

    parser.add_argument('--train_DB', type=list, default=['ESBD'], choices=['SSBD', 'ESBD'])
    parser.add_argument('--test_DB', type=list, default=['SSBD'], choices=['SSBD', 'ESBD'])
    parser.add_argument('--crop_ROI', type=str, default='whole', choices=['whole'])

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

        output = model(rgb)
        _, preds = torch.max(output, 1)
        loss = criteria(output, label)

        # Backward
        optimizer.zero_grad()

        # Backward into the branch
        if is_training:
            loss.backward()
            optimizer.step()

    return loss, preds


def _train(model, criteria, optimizer, dataloaders, device=None, num_epochs=200):

    log_dir = args.checkpointdir + '/logs/logs_classifier/' + args.model_name
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

                input_data = [data[n].to(device) for n in ['rgb', 'label']]
                loss, preds = step(model, input_data, optimizer, criteria, is_training)

                # Update statistics
                batch_size = input_data[0].size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == input_data[-1].data)
                ndata = ndata + batch_size

                # save loss
                iter_adder_visual(loss.item())

                if (iter_idx + 1) % args.print_freq == 0 and is_training:
                    lr = check_lr(optimizer)
                    print("Epoch: %03d/%03d Iter: %4d/%4d  Loss: %7.4f  LR:%.10f" % (
                            epoch, args.epochs, iter_idx + 1, max_iter,
                            iter_adder_visual.average(),  lr))

                    writer.add_scalar('Loss_{}/iter'.format('train'), iter_adder_visual.average(),
                                      iter_idx + (epoch - 1) * max_iter)

                    iter_adder_visual.reset()

            epoch_loss = running_loss / ndata
            epoch_acc = running_corrects.double() / ndata
            print('Loss_epoch: {}, Accuracy_epoch: {}'.format(epoch_loss, epoch_acc))

            # tensorboard update
            writer.add_scalar('Acc_{}/epoch'.format(phase), epoch_acc, epoch)
            writer.add_scalar('Loss_{}/epoch'.format(phase), epoch_loss, epoch)
            writer.flush()


        if epoch == num_epochs:
            filename = (args.checkpointdir + '/logs/logs_classifier/' + args.model_name + '/model_%d.checkpoint' % epoch)
            torch.save(model.state_dict(), filename)
            print('Saving ' + filename)

            print('testing with best test model')
            test_acc = _test_model(model, dataloaders, args, device, 'test')
            fp = open(os.path.join('checkpoints/logs/logs_classifier', args.model_name, 'Results_test.txt'), 'a')
            fp.write('Epoch:Final ')
            fp.write('test_Accuracy: {:.3f}, recall: {:.3f}, precision: {:.3f}, f_score: {:.3f} \n'
                .format(test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
            fp.close()
            print('test_Accuracy: {:.3f}'.format(test_acc[0]))


def _test(model, dataloaders, phase, device=None):
    model.train(False)
    running_corrects = 0
    label_pred = []
    label_gt = []

    # Iterate over data
    ndata = 0
    with torch.no_grad():
        for iter_idx, data in enumerate(tqdm(dataloaders[phase])):

            rgb, label = [data[n].to(device) for n in ['rgb', 'label']]
            output = model(rgb)
            preds = torch.argmax(output, dim=-1)

            ndata += rgb.size(0)

            label_pred.append(preds.cpu().detach().numpy())
            label_gt.append(label.cpu().detach().numpy())



    precision, recall, fscore, _ = score(label_gt, label_pred, average="weighted")
    acc = running_corrects.double() / ndata
    print('------Final accuracy on segment-level action classification------')
    print('accuracy: {}'.format(acc))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))

    print(confusion_matrix(label_gt, label_pred))
    print(classification_report(label_gt, label_pred, target_names=['ArmFlapping', 'Spinning', 'HeadBanging']))

    acc_list = [acc, recall, precision, fscore]

    return acc_list


def _test_model(model, dataloaders, args, device, phase):
    #if args.train is False:
    if phase == 'test':

        testmodel_filename = (args.checkpointdir + '/logs/logs_classifier/' + args.model_name + '/model_50.checkpoint')

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

    return test_model_acc


def _train_model(model, dataloaders, args, device):


    criteria = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    device = torch.device("cuda:0" if use_gpu else "cpu") # cuda:1
    dataloaders = get_dataloaders(args)

    num_class = max(dataloaders['train'].dataset.labels) + 1

    if args.backbone == 'VST':
        model = VideoSwinTransformerModel(num_classes=num_class) # -->> hugging face나 timm에 좀 쉽게 돼있는 모델 찾아보자.
    elif args.backbone == 'VideoMAE':
        model = VideoMAE(num_classes=num_class)
    elif args.backbone == 'I3D':
        model = I3D(num_classes=num_class)

    ## Params (MB)
    num_of_parameters = sum(map(torch.numel, model.parameters()))
    print("Total memory of paramter(MB):{}".format(num_of_parameters * 4 / 1024 / 1024.0))
    print("Total number of parameters (M): {:.2f}M".format(num_of_parameters / 1e6))

    ## GFLOPs
    # 모델을 GPU로 이동
    # model = model.cuda()
    # model.eval()
    #
    # input = torch.randn(1, 3, 16, 224, 224)
    # input = input.cuda()
    #
    # flops, parmas = profile(model, inputs=(input,))
    # flops /= (1000 * 1000 * 1000)
    # print("GFLOPs: {:.2f}B".format(flops))

    #flops_, params_ptflops = get_model_complexity_info(model, (3, 16, 224, 224), as_strings=True, print_per_layer_stat=True)

    if args.train:

        train_path = os.path.join('checkpoints/logs/logs_classifier', args.model_name)
        Path(train_path).mkdir(parents=True, exist_ok=True)
        train_args_dir = os.path.join(train_path, 'train_args.txt')

        with open(train_args_dir, 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        print("Saving train args")
        print("Training Proposed network")

        _train_model(model, dataloaders, args, device)
    else:

        with open(os.path.join('checkpoints/logs/logs_classifier', args.model_name, 'test_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        print("Saving test args")
        print("Evaluating single model...")

        test_acc = _test_model(model, dataloaders, args, device, 'test') #'test
        print('test_Accuracy: {:.3f}\n'.format(test_acc[0]))

        fp = open(os.path.join('checkpoints/logs/logs_classifier', args.model_name, 'Results_test.txt'), 'a')
        fp.write('Epoch:best ')
        fp.write('test_Accuracy: {:.3f}, recall: {:.3f}, precision: {:.3f}, f_score: {:.3f} \n'
                 .format(test_acc[0], test_acc[1], test_acc[2], test_acc[3]))
        fp.close()
