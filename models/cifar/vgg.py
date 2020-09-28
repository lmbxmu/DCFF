import torch
import torch.nn as nn

from .fuse_modules import *


cfg = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, cprate, num_class=10):
        super(VGG, self).__init__()

        self.features = self._make_layers(cfg[vgg_name], cprate)
        cout = int(512*(1-cprate[-1]))
        self.classifier = nn.Linear(cout, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _make_layers(self, cfg, cprate):
        layers = []
        in_channels = 3
        conv_layerid = 0
        for layeri, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                cout = int(x*(1-cprate[conv_layerid]))
                layers += [FuseConv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(cout),
                           nn.ReLU(inplace=True)]
                in_channels = cout
                conv_layerid += 1

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)


