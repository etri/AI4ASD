# -*- coding: utf-8 -*- 

""" 
   * Source: libFER.cda_trainer.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI

"""

import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

from libFER.arg_parser_cda import argument_parser_cda
from libFER.model_selector import select_model
from libFER.model_cda import CrossDatasetAdaptation, CrossDatasetAdaptation_Relu, CrossDatasetAdaptation_Conv
from libFER.data_loader import ImageList
from libFER.custom_loss import AdaptiveCrossEntropyLoss
import libFER.measures as measures
import libFER.utils as utils

ds_alpha = 0.0
ace_rate = 0.0

args = argument_parser_cda()


def cda_train():

    """cda_train function

    Note:   Train function using a designated trained pth file.
            Using arg_parser_cda.py, parameters for train are listed.

    Arguments: 
        
    Returns:

    """

    global args
    
    fe_model = select_model(args.model, False, True)

    if args.classifier_type == '1-fc':
        model = CrossDatasetAdaptation(fe_model, args.num_classes, args.num_ds_classes, with_label=False)
    elif args.classifier_type == '3-fc':
        model = CrossDatasetAdaptation_Relu(fe_model, args.num_classes, args.num_ds_classes, with_label=False)
    elif args.classifier_type == 'conv':
        model = CrossDatasetAdaptation_Conv(fe_model, args.num_classes, args.num_ds_classes, with_label=False)

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate[0],
                                    momentum=args.momentum,
                                    weight_decay = args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate[0], 
                                     weight_decay = args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print('=> loaded checkpoint "{}" (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path, datalist_filename=args.train_list,
                  transform=transforms.Compose([
                      transforms.Resize((100, 100)),
                      transforms.RandomCrop((100, 100), padding=(16, 16)),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path, datalist_filename=args.val_list,
                  transform=transforms.Compose([
                      transforms.Resize((100, 100)),
                      transforms.ToTensor(),
                      ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.emo_loss == 'cce':
        criterion = nn.CrossEntropyLoss()
    elif args.emo_loss =='ace':
        criterion = AdaptiveCrossEntropyLoss()

    if args.ds_loss == 'cce':
        criterion_ds = nn.CrossEntropyLoss()
    elif args.ds_loss == 'ace':       
        criterion_ds = AdaptiveCrossEntropyLoss()
   
    if args.cuda:
        criterion.cuda()
        criterion_ds.cuda()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open(args.save_path + '/' + 'argument.txt', 'w') as f:
        print(vars(args))
        for (key, value) in vars(args).items():
            f.write(str(key) + ' : ' + str(value) + '\n')      

    validate(val_loader, model, criterion, criterion_ds)
    global ds_alpha
    global ace_rate

    accuracy_list=[]

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        print("#### [ TRAIN ] ####")
        lr = utils.rate_scheduler(epoch, args.lr_boundary, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        ds_alpha = utils.rate_scheduler(epoch, args.boundary, args.ds_alpha)
        ace_rate = utils.rate_scheduler(epoch, args.boundary, args.ace_rate)
            
        train_loss = train(train_loader, model, criterion, criterion_ds, optimizer, epoch)
        print('@# train_loss = ', train_loss)
        print("#### [ EVAL ] ####")
        prec1 = validate(val_loader, model, criterion, criterion_ds)
        accuracy_list.append(prec1.item())
        
        save_name = args.save_path + '/' + args.model + '_' 
        save_name += str(epoch+1) + '_%.2f'%prec1.item() + '_checkpoint.pth.tar'
        torch.save({
            'epoch': epoch + 1,
            'model' : args.model,
            'state_dict': model.state_dict(),
            'prec1': prec1,
            }, save_name)
        epoch_time = time.time()- epoch_start
        estimated_total_time = epoch_time * args.epochs
        _day = estimated_total_time//60//60//24
        _hour = (estimated_total_time - _day*60*60*24)//60//60
        _min = (estimated_total_time - _day*60*60*24 - _hour*60*60)//60
        print("** Time for an epoch: ", epoch_time, "secs")
        print("** Estimated total time: ", _day, ' days ', _hour, ' hours ', _min, ' mins')
    max_accuracy = max(accuracy_list)
    np.savetxt(args.save_path  + '/' + 'max_accuracy_' + str(max_accuracy) + '.txt', np.array([max_accuracy]))


def train(train_loader, model, criterion, criterion_ds, optimizer, epoch):

    """train function

    Note:   Train function

    Arguments: 
        train_loader (torch.utils.data.DataLoader): data loader
        model: pytorch neural net model
        criterion: loss object for emotion classification
        criterion_ds: loss object for dataset classification
        optimizer: optimizer object
        epoch: current epoch

    Returns:
        losses.avg: average loss value

    """

    batch_time  = utils.AverageMeter()
    data_time   = utils.AverageMeter()
    emo_losses      = utils.AverageMeter()
    ds_losses = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1        = utils.AverageMeter()
    top1_ds = utils.AverageMeter()
    cfm = utils.AverageMeter()
    cfm_ds = utils.AverageMeter()
    softlabel_mean = utils.AverageMeter()
    softlabel_std = utils.AverageMeter()

    model.train()
    end = time.time()

    for i, (input, label, ds_label) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        one_hot_label = torch.zeros((input.size(0), args.num_classes), dtype=torch.float)
        one_hot_label[torch.arange(input.size(0)), label] = 1.0
        one_hot_label_ds = torch.zeros((input.size(0), args.num_ds_classes), dtype=torch.float)
        one_hot_label_ds[torch.arange(input.size(0)), ds_label] = 1.0
       
        if args.cuda:
            input = input.cuda()
            label = label.cuda()
            ds_label = ds_label.cuda()
            one_hot_label = one_hot_label.cuda()
            one_hot_label_ds = one_hot_label_ds.cuda()

        input_var = torch.autograd.Variable(input)
        label_var = torch.autograd.Variable(label)
        ds_label_var = torch.autograd.Variable(ds_label)
        one_hot_label_var = torch.autograd.Variable(one_hot_label)
        one_hot_label_ds_var = torch.autograd.Variable(one_hot_label_ds)

        output, ds_output = model(input_var, ds_alpha, one_hot_label_var)     
            
        batch_cfm = measures.get_confusion_matrix_3(label, torch.argmax(output, dim=1), args.num_classes)
        cfm.update(batch_cfm, 1)
        batch_cfm = batch_cfm.float()
        batch_ds_cfm = measures.get_confusion_matrix_3(ds_label, torch.argmax(ds_output, dim=1), args.num_ds_classes)
        cfm_ds.update(batch_ds_cfm, 1)
        batch_ds_cfm = batch_ds_cfm.float()
        if args.cuda:
            batch_cfm = batch_cfm.cuda()
            batch_ds_cfm = batch_ds_cfm.cuda()
            
        if args.emo_loss == 'cce':
            emo_loss = criterion(output, label_var)
        elif args.emo_loss == 'ace':
            emo_loss = criterion(output, one_hot_label_var, batch_cfm, ace_rate)
        if args.ds_loss == 'cce':
            ds_loss = criterion_ds(ds_output, ds_label_var)
        elif args.ds_loss == 'ace':
            ds_loss = criterion_ds(ds_output, one_hot_label_ds_var, batch_ds_cfm, ace_rate)                               

        loss = emo_loss + ds_loss 

        prec1, _ = measures.accuracy(output.data, label, topk=(1, 5))
        prec1_ds, _  = measures.accuracy(ds_output, ds_label, topk=(1, 3))
        emo_losses.update(emo_loss.data, input.size(0))
        ds_losses.update(ds_loss.data, input.size(0))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top1_ds.update(prec1_ds, input.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        batch_time.update(time.time()-end)
        end=time.time()

        if i%args.print_freq == 0:
            print('# Epoch: [{0}][{1}/{2}]\n'
                  '# BatchTime(avg): {batch_time.val:.3f} sec ({batch_time.avg:.3f} sec)'
                  '>> DataLoad(avg)={data_time.avg:.3f} \n'
                  '# Loss {loss.val:.4f} >> EmoLoss={emo_loss.val:.4f}, DSLoss={ds_loss.val:.4f}\n'
                  '# Accuracy(avg): Emotion={top1.val:.3f}%({top1.avg:.3f}%)'
                  '\tDS={top1_ds.val:.3f}%({top1_ds.avg:.3f}%)\n'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, 
                      emo_loss = emo_losses, ds_loss=ds_losses, 
                      top1=top1, top1_ds=top1_ds, cfm=cfm, cfm_ds=cfm_ds))

    return losses.avg
            

def validate(val_loader, model, criterion, criterion_ds):

    """train function

    Note:   Train function

    Arguments: 
        val_loader (torch.utils.data.DataLoader): data loader
        model: pytorch neural net model
        criterion: loss object for emotion classification
        criterion_ds: loss object for dataset classification

    Returns:
        top1.avg: average value for top1 accuracies

    """

    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top1_ds = utils.AverageMeter()
    top5 = utils.AverageMeter()
    cfm = utils.AverageMeter()
    cfm_ds = utils.AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, label, ds_label) in enumerate(val_loader):

        one_hot_label = torch.zeros((input.size(0), args.num_classes), dtype=torch.float)
        one_hot_label[torch.arange(input.size(0)), label] = 1.0
        one_hot_label_ds = torch.zeros((input.size(0), args.num_ds_classes), dtype=torch.float)
        one_hot_label_ds[torch.arange(input.size(0)), ds_label] = 1.0

        if args.cuda:
            input = input.cuda()
            label = label.cuda()
            ds_label = ds_label.cuda()
            one_hot_label = one_hot_label.cuda()
            one_hot_label_ds = one_hot_label_ds.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            label_var = torch.autograd.Variable(label)
            ds_label_var = torch.autograd.Variable(ds_label)
            one_hot_label_var = torch.autograd.Variable(one_hot_label)
            one_hot_label_ds_var = torch.autograd.Variable(one_hot_label_ds)

        output, ds_output = model(input_var, ds_alpha, one_hot_label_var)
  
        batch_cfm = measures.get_confusion_matrix_3(label, torch.argmax(output, dim=1), args.num_classes)
        cfm.update(batch_cfm, 1)
        batch_cfm = batch_cfm.float()
        batch_ds_cfm = measures.get_confusion_matrix_3(ds_label, torch.argmax(ds_output, dim=1), args.num_ds_classes)
        cfm_ds.update(batch_ds_cfm, 1)
        batch_ds_cfm = batch_ds_cfm.float()

        if args.cuda:
            batch_cfm = batch_cfm.cuda()
            batch_ds_cfm = batch_ds_cfm.cuda()

        if args.emo_loss == 'cce':
            emo_loss = criterion(output, label_var)
        elif args.emo_loss =='ace':
            emo_loss = criterion(output, one_hot_label_var, batch_cfm, ace_rate)

        if args.ds_loss =='cce':
            ds_loss = criterion_ds(ds_output, ds_label_var)
        elif args.ds_loss == 'ace':
            ds_loss = criterion_ds(ds_output, one_hot_label_ds_var, batch_ds_cfm, ace_rate)

        loss = emo_loss + ds_loss 
        
        # measure accuracy and record loss
        prec1, prec5 = measures.accuracy(output.data, label, topk=(1, 5))
        prec1_ds, _ = measures.accuracy(ds_output.data, ds_label, topk=(1, 3))

        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top1_ds.update(prec1_ds, input.size(0))

    print('## Accuracy: {1:.2f}%, Average loss {0:.4f}\n'
          '## DS Accuracy: {3:.2f}%\n'
          '## Confusion matrix: \n{2}\n'
          '## DS Confusion matrix: \n{4}\n'.format(losses.avg, top1.avg, cfm.sum, top1_ds.avg, cfm_ds.sum))
    return top1.avg
      

