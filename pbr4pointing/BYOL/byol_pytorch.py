"""
   * Source: byol_pytorch.py
   * License: PBR4AI License (Dual License)
   * Modified by Cheol-Hwan Yoo <ch.yoo@etri.re.kr>
   * Date: 21 Aug. 2023, ETRI
   * Copyright 2023. ETRI all rights reserved.
"""

import copy
import random
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from functools import wraps
from torchvision import transforms as T


def default(val, def_val):

    """default function

    Note: function for default

    """

    return def_val if val is None else val

def flatten(t):

    """flatten function

    Note: function for flatten

    """

    return t.reshape(t.shape[0], -1)

def singleton(cache_key):

    """singleton function

    Note: function for singleton

    """

    def inner_fn(fn):

        """inner_fn function

        Note: function for inner_fn

        """

        @wraps(fn)
        def wrapper(self, *args, **kwargs):

            """wrapper function

            Note: function for wrapper

            """

            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def get_module_device(module):

    """get_module_device function

    Note: function for get_module_device

    """

    return next(module.parameters()).device

def set_requires_grad(model, val):

    """set_requires_grad function

    Note: function for set_requires_grad

    """

    for p in model.parameters():
        p.requires_grad = val


def loss_fn(x, y):

    """loss_fn function

    Note: function for loss_fn

    """

    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class RandomApply(nn.Module):

    """RandomApply class

    Note: class for RandomApply

    """

    def __init__(self, fn, p):

        """ __init__ function for RandomApply class

        """

        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):

        """ forward function for RandomApply class

        """

        if random.random() > self.p:
            return x
        return self.fn(x)


class EMA():

    """ exponential moving average(EMA) class

    Note: class for exponential moving average(EMA)

    """

    def __init__(self, beta):

        """ __init__ function for EMA class

        """

        super().__init__()
        self.beta = beta

    def update_average(self, old, new):

        """ update_average function for EMA class

        """

        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):

    """update_moving_average function

    Note: function for update_moving_average

    """

    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def MLP(dim, projection_size, hidden_size=4096):

    """MLP function

    Note: MLP function for projector and predictor

    """

    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=4096):

    """SimSiamMLP function

    Note: function for SimSiamMLP

    """

    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):

    """ NetWrapper class

    Note: class for NetWrapper

    """

    def __init__(self, net, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False):

        """ __init__ function for NetWrapper class

        """

        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):

        """ _find_layer function for NetWrapper class

        """

        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):

        """ _hook function for NetWrapper class

        """

        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):

        """ _hook _register_hook for NetWrapper class

        """

        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):

        """ _get_projector for NetWrapper class

        """

        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):

        """ get_representation for NetWrapper class

        """

        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):

        """ forward for NetWrapper class

        """

        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


class BYOL(nn.Module):

    """ BYOL class

    Note: main class for BYOL

    """

    def __init__(
        self,
        net,
        image_size,
        pre_class_dim,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True
    ):

        """ __init__ for BYOL class

        """

        super().__init__()
        self.net = net

        # default SimCLR augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size
                                         , layer=hidden_layer, use_simsiam_mlp=not use_momentum)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        ## final logit layer
        self.classifier = nn.Linear(in_features=pre_class_dim, out_features=2)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)


    @singleton('target_encoder')
    def _get_target_encoder(self):

        """ _get_target_encoder for BYOL class

        """

        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):

        """ reset_moving_average for BYOL class

        """

        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):

        """ update_moving_average for BYOL class

        """

        assert self.use_momentum, 'you do not need to update the moving average, ' \
                                  'since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, x1, x2):

        """ forward for BYOL class

        """

        if self.training == True:
            online_pred_one, latent_feature_one = self.online_encoder(x1)
            online_pred_two, _ = self.online_encoder(x2)

            online_pred_one = self.online_predictor(online_pred_one)
            online_pred_two = self.online_predictor(online_pred_two)

            with torch.no_grad():
                target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
                target_proj_one, _ = target_encoder(x1)
                target_proj_two, _ = target_encoder(x2)
                target_proj_one.detach_()
                target_proj_two.detach_()

            loss_one = loss_fn(online_pred_one, target_proj_two.detach())
            loss_two = loss_fn(online_pred_two, target_proj_one.detach())

            loss = loss_one + loss_two

        else:
            latent_feature_one = self.online_encoder(x1, return_projection = False)
            #latent_feature_two = self.online_encoder(x2, return_projection = False)
            loss = np.array([0])

        #### adding logit layer
        logits1 = self.classifier(latent_feature_one)
        #logits2 = self.classifier(online_proj_two_latent)


        return logits1, loss.mean()