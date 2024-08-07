#!/usr/bin/env python3

import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights


class Classifier(nn.Module):
    def __init__(self, num_ftrs, n_class, drop_out=0.5, hidden_dim=8):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=drop_out),
            nn.Linear(hidden_dim, n_class),
        )

    def forward(self, x):
        return self.classifier(x)


def get_model(fix_depth=1, backbone='resnet18', n_class=2, drop_out=0.5, hidden_dim=8):
    """

    :param fix_depth: choice from range(1, 6)
    :param backbone: [resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2,
                        inception_v3, vgg11_bn, vgg13_bn, vgg16_bn]
    :param n_class: output number
    :param drop_out: in (0, 1), invalid if hidden_dim is 0
    :param hidden_dim: if 0 then no hidden layer
    :return:
    """

    """

    :param n_class:
    :param backbone: 
    :param fix_depth:
    :return:
    """

    fun = getattr(models, backbone)
    model = fun(weights=ResNet18_Weights.IMAGENET1K_V1)

    # fronze layers from fix_depth+1 to end
    if 'resnet' in backbone:
        freeze_layers_4resnet(model, fix_depth)
    if 'inception' in backbone:
        freeze_layers_4goolenet(model, fix_depth)
    if 'vgg' in backbone:
        freeze_layers_4vgg(model, fix_depth, backbone)

    # change classifier
    if 'resnet' in backbone or 'inception' in backbone:
        num_ftrs = model.fc.in_features
        if hidden_dim == 0:
            model.fc = nn.Linear(num_ftrs, n_class)
        else:
            model.fc = Classifier(num_ftrs, n_class, drop_out, hidden_dim)
    if 'vgg' in backbone:
        for idx, (key, value) in enumerate(model.classifier._modules.items()):
            if idx == len(model.classifier._modules) - 1:
                num_ftrs = model.classifier._modules[key].in_features
                if hidden_dim == 0:
                    model.classifier._modules[key] = nn.Linear(num_ftrs, n_class)
                else:
                    model.classifier._modules[key] = Classifier(num_ftrs, n_class, drop_out, hidden_dim)

    return model


def freeze_layers_4resnet(model, fix_depth):
    if fix_depth > 0:
        for param in model.parameters():
            param.requires_grad = False

        for idx in range(fix_depth + 1, 5):
            layer = 'layer{0}'.format(idx)
            for param in getattr(model, layer).parameters():
                param.requires_grad = True


def freeze_layers_4goolenet(model, fix_depth):
    def unfronze_layers(model, layers):
        for layer in layers:
            for param in getattr(model, layer).parameters():
                param.requires_grad = True

    if fix_depth > 0:
        for param in model.parameters():
            param.requires_grad = False

        if fix_depth == 3:
            unfronze_layers(model, ['Mixed_7a', 'Mixed_7b', 'Mixed_7c'])

        if fix_depth == 2:
            unfronze_layers(model, ['Mixed_7a', 'Mixed_7b', 'Mixed_7c'])
            unfronze_layers(model, ['Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e'])

        if fix_depth == 1:
            unfronze_layers(model, ['Mixed_7a', 'Mixed_7b', 'Mixed_7c'])
            unfronze_layers(model, ['Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e'])
            unfronze_layers(model, ['Mixed_5b', 'Mixed_5c', 'Mixed_5d'])


def freeze_layers_4vgg(model, fix_depth, backbone):
    def unfronze_layer(model, layers):
        for layer in layers:
            for param in model.features._modules[layer].parameters():
                param.requires_grad = True

    if fix_depth > 0:
        for param in model.parameters():
            param.requires_grad = False

        layers_dict = {
            'vgg11_bn': [(4, 29), (8, 29), (15, 29), (22, 29)],
            'vgg13_bn': [(7, 35), (14, 35), (21, 35), (28, 35)],
            'vgg16_bn': [(7, 35), (14, 35), (21, 35), (28, 35)],
            'vgg19_bn': [(7, 53), (14, 53), (27, 53), (40, 53)],
        }
        unfronze_layer(model, [str(i) for i in range(*layers_dict[backbone][fix_depth-1])])

