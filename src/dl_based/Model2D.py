#!/usr/bin/env python3

import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, fix_depth=1):
        super(Model, self).__init__()

        self.fix_depth = fix_depth

    def model_define(self, backbone='resnet18', n_class=2):
        """

        :param n_class:
        :param backbone: [resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2,
                            inception_v3, vgg11_bn, vgg13_bn, vgg16_bn]
        :param fix_depth:
        :return:
        """

        fun = getattr(models, backbone)
        self.model = fun(weights='IMAGENET1K_V1')

        # fronze layers from fix_depth+1 to end
        if 'resnet' in backbone:
            self.freeze_layers_4resnet()
        if 'inception' in backbone:
            self.freeze_layers_4goolenet()
        if 'vgg' in backbone:
            self.freeze_layers_4vgg(backbone)

        # change classifier
        if 'resnet' in backbone or 'inception' in backbone:
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, n_class)
        if 'vgg' in backbone:
            for idx, (key, value) in enumerate(self.model.classifier._modules.items()):
                if idx == len(self.model.classifier._modules) - 1:
                    num_ftrs = self.model.classifier._modules[key].in_features
                    self.model.classifier._modules[key] = nn.Linear(num_ftrs, n_class)

        return self.model

    def freeze_layers_4resnet(self):
        if self.fix_depth > 0:
            for param in self.model.parameters():
                param.requires_grad = False

            for idx in range(self.fix_depth + 1, 5):
                layer = 'layer{0}'.format(idx)
                for param in getattr(self.model, layer).parameters():
                    param.requires_grad = True

    def freeze_layers_4goolenet(self):
        def unfronze_layers(model, layers):
            for layer in layers:
                for param in getattr(model, layer).parameters():
                    param.requires_grad = True

        fix_depth = self.fix_depth
        model = self.model

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

    def freeze_layers_4vgg(self, backbone):
        def unfronze_layer(model, layers):
            for layer in layers:
                for param in model.features._modules[layer].parameters():
                    param.requires_grad = True

        fix_depth = self.fix_depth
        model = self.model

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


if __name__ == '__main__':
    model = Model().model_define(backbone='resnet18')
    print(model)