""" 
   * Source: libFER.model_selector.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI
   * Copyright 2022. ETRI all rights reserved. 

"""

from libFER.models import model_alexnet
from libFER.models import model_resnet
from libFER.models import model_vgg
from libFER.models import model_densenet
from libFER.models import model_mobilenet
from libFER.models import model_squeezenet
from libFER.models import model_googlenet
from libFER.models import model_inception
from libFER.models import model_shufflenetv2
from libFER.models import model_mnasnet


def select_model(model_name, pretrained = False, progress = True, **kwargs):

    """select_model function

    Note: model selection function

    Arguments: 
        model_name (str): model name
        pretrained (bool): usage of pretrained or not
        progress (bool): showing progress information or not
        kwargs: other arguments

    Returns:
        model: Pytorch Neural Net model

    """


    if model_name == 'resnet18':
        model = model_resnet.resnet18(pretrained, progress, **kwargs)
    elif model_name == 'resnet34':
        model = model_resnet.resnet34(pretrained, progress, **kwargs)
    elif model_name == 'resnet50':
        model = model_resnet.resnet50(pretrained, progress, **kwargs)
    elif model_name == 'resnet101':
        model = model_resnet.resnet101(pretrained, progress, **kwargs)
    elif model_name == 'resnet152':
        model = model_resnet.resnet152(pretrained, progress, **kwargs)
    elif model_name == 'resnext50_32x4d':
        model = model_resnet.resnext50_32x4d(pretrained, progress, **kwargs)
    elif model_name == 'resnext101_32x8d':
        model = model_resnet.resnext101_32x8d(pretrained, progress, **kwargs)
    elif model_name == 'wide_resnet50_2':
        model = model_resnet.wide_resnet50_2(pretrained, progress, **kwargs)
    elif model_name == 'wide_resnet101_2':
        model = model_resnet.wide_resnet101_2(pretrained, progress, **kwargs)       
    elif model_name == 'alexnet':
        model = model_alexnet.alexnet(pretrained, progress, **kwargs)
    elif model_name =='vgg19_bn':
        model = model_vgg.vgg19_bn(pretrained, progress, **kwargs)
    elif model_name =='densenet121':
        model = model_densenet.densenet121(pretrained, progress, **kwargs)
    elif model_name =='mobilenet_v2':
        model = model_mobilenet.mobilenet_v2(pretrained, progress, **kwargs)
    elif model_name =='squeezenet1_1':
        model = model_squeezenet.squeezenet1_1(pretrained, progress, **kwargs)
    elif model_name =='googlenet':
        model = model_googlenet.googlenet(pretrained, progress, **kwargs)
    elif model_name =='inception_v3':
        model = model_inception.inception_v3(pretrained, progress, **kwargs)
    elif model_name == 'LightCNN_29Layers_v2':
        model = model_lightcnn.LightCNN_29Layers_v2(**kwargs)
    elif model_name == 'shufflenet_v2_x2_0':
        model = model_shufflenetv2.shufflenet_v2_x2_0(pretrained, progress, **kwargs)
    elif model_name == 'mnasnet1_3':
        model = model_mnasnet.mnasnet1_3(pretrained, progress, **kwargs) 
        
    return model

