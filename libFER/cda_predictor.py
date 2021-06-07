""" 
   * Source: libFER.cda_predictor.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI

"""

import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import cv2
from PIL import Image

from libFER.arg_parser_cda import argument_parser_cda
from libFER.model_selector import select_model
from libFER.model_cda import CrossDatasetAdaptation, CrossDatasetAdaptation_Relu, CrossDatasetAdaptation_Conv

args = argument_parser_cda()


def cda_predict(cv_img):

    """cda_predict function

    Note:   Emotion prediction function using a designated trained pth file.
            Using arg_parser_cda.py, parameters for prediction are listed.

    Arguments: 
        cv_img (opencv image): image to predict emotion

    Returns:
        output (numpy): predicted emotion label

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

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print('=> loaded checkpoint "{}" (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(cv_img)


    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])                   
    transform = transforms.Compose([
        transforms.Resize([100, 100]),
        transforms.ToTensor(),
        normalize])
    face_tensor = transform(face_pil)
    face_tensor = face_tensor.view(1, 3, 100, 100)
    if args.cuda:
        face_tensor = face_tensor.cuda()

    output, _ = model(face_tensor)
    output = torch.argmax(output, dim=1)

    return output.cpu().numpy()
