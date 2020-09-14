import torch.nn as nn
from collections.abc import Iterable   # drop `.abc` with Python 2.7 or lower

def freeze_layer(layer):
    parameters = list(layer.parameters())
    for param in parameters:
        param.requires_grad = False

def freeze_network(network):
    for name, p in network.named_parameters():
        # if "network2" in name:
        p.requires_grad = False
    #
    # if not isinstance(network, Iterable):
    #     if not isinstance(network, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
    #         freeze_layer(network)
    #     return
    #
    # for block in network:
    #     freeze_network(block)