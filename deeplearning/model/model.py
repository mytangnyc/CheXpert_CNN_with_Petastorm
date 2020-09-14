import torch.nn as nn
import torch.nn.functional as F
import torch
from deeplearning.base import BaseModel
import torchvision.models.densenet as DenseNet

from deeplearning.model.ChexpertModelUtil.backbone.densenet import (densenet121, densenet169, densenet201)
from deeplearning.model.ChexpertModelUtil.helper_layers import Flatten, AdaptiveConcatPool2d
from deeplearning.model.ChexpertModelUtil.utils import freeze_network
BACKBONES = {

             'densenet121': densenet121,
             'densenet169': densenet169,
             'densenet201': densenet201,
             }


BACKBONES_TYPES = {
                   'densenet121': 'densenet',
                   'densenet169': 'densenet',
                   'densenet201': 'densenet',
                   }


class ChexpertNext1(BaseModel):
    def __init__(self, **kwargs):
        super(ChexpertNext1, self).__init__()
        self.cfg = kwargs
        self.backbone = BACKBONES[self.cfg["backbone"]](pretrained=self.cfg["pretrained"], cutoff=self.cfg["backbone_cutoff"])
        if self.cfg["freeze_pretrain"]:
            freeze_network(self.backbone)

        self.head_input_size = 1024 * 2
        self.head_output_size = 14
        self.classifier = self._init_classifier()

    def forward(self, x):
        out = self.backbone(x)
        return self.classifier(out)

    def _init_classifier(self):
        layers = [self.head_input_size, 512, self.head_output_size]
        dropout = [0.25, 0.5]
        activation = [nn.ReLU(inplace=True)] + [None]
        pool = AdaptiveConcatPool2d()
        combined_layer = [pool, Flatten()]
        for in_sz, out_sz, drop_rate, acti in zip(layers[:-1], layers[1:], dropout, activation):
            block = [nn.BatchNorm1d(in_sz), nn.Dropout(drop_rate), nn.Linear(in_sz, out_sz)]
            if acti: block.append(acti)

            combined_layer.extend(block)
        return nn.Sequential(*combined_layer)



class SimpleChexpertNet(BaseModel):
    """
    Simple Chexpert Net for iteration. The image went through a base
    """
    def __init__(self, **kwargs):
        super(SimpleChexpertNet, self).__init__()
        self.cfg = kwargs
        self.backbone = BACKBONES[self.cfg["backbone"]](self.cfg)
        self.feature_extraction = nn.Linear(1024, 1024)
        self.expand = 1
        self._init_classifier()

    def forward(self, x):
        out = self.backbone(x)
        # print (out.size())
        feature = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        logit_maps = [] # array that stores the result for each class
        for index in range(self.cfg["num_classes"]):
            classifier = getattr(self, "fc_{}".format(index))
            logit = classifier(feature)
            logit_maps.append(logit)
        cat_logits = torch.cat(logit_maps, 1)
        # print (cat_logits[0])
        return cat_logits

    def _init_classifier(self):
        for index in range(self.cfg["num_classes"]):
            # this is setting up the
            num_class = 1
            if BACKBONES_TYPES[self.cfg["backbone"]] == 'densenet':
                setattr(
                    self,
                    "fc_{}".format(index),
                    nn.Linear(
                        1024,
                        1,  # out feature classification title
                        bias=True))

