import torch
import torch.nn as nn

from .fuse_modules import FuseConv2d


cfg = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

#*================================
#* fused model
class Fused_VGG(nn.Module):
    def __init__(self, vgg_name, cprate, num_class=10):
        super(Fused_VGG, self).__init__()

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
        cur_fconvid = 0
        for layeri, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                cout = int(x*(1-cprate[cur_fconvid]))
                layers += [FuseConv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1, real_cout=cout),
                           nn.BatchNorm2d(cout),
                           nn.ReLU(inplace=True)]
                in_channels = cout
                cur_fconvid += 1

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)



#*================================
#* compact model
class Compact_VGG(nn.Module):
    def __init__(self, vgg_name, cprate, num_class=10):
        super(Compact_VGG, self).__init__()

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
        cur_fconvid = 0
        for layeri, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                cout = int(x*(1-cprate[cur_fconvid]))
                layers += [nn.Conv2d(in_channels, cout, kernel_size=3, padding=1),
                           nn.BatchNorm2d(cout),
                           nn.ReLU(inplace=True)]
                in_channels = cout
                cur_fconvid += 1

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)




#*================================
#* origin model
class OriginVGG(nn.Module):
    def __init__(self, vgg_name, num_class=10):
        super(OriginVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

