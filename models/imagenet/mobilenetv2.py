import torch
import torch.nn as nn
import math
import pdb

from .fuse_modules import FuseConv2d
# from fuse_modules import FuseConv2d

cur_convid = -1
last_cout = -1
cur_cout = -1

#*================================
#* Fused model
def fused_conv_bn(inp, oup, stride, layers_cprate):
    global cur_convid, last_cout, cur_cout

    last_cout = cur_cout
    cur_cout = int(oup * (1-layers_cprate[cur_convid]))
    cur_convid += 1
    return nn.Sequential(
        # nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        FuseConv2d(last_cout, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cur_cout),
        nn.ReLU6(inplace=True)
    )

def fused_conv_1x1_bn(inp, oup, layers_cprate):
    global cur_convid, last_cout, cur_cout
    #*
    last_cout = cur_cout
    cur_cout = int(oup * (1-layers_cprate[cur_convid]))
    cur_convid += 1
    return nn.Sequential(
        # nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        FuseConv2d(last_cout, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(cur_cout),
        nn.ReLU6(inplace=True)
    )

def fused_make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class FusedInvertedResidual(nn.Module):
    
    def __init__(self, inp, oup, stride, expand_ratio, layers_cprate):
        global cur_convid, last_cout, cur_cout
        super(FusedInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            last_cout = cur_cout
            cur_cout = last_cout

            last_cout2 = cur_cout
            cur_cout2 = int(oup * (1-layers_cprate[cur_convid]))
            cur_convid += 1
            self.conv = nn.Sequential(
                # dw
                # nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.Conv2d(last_cout, cur_cout, kernel_size=3, stride=stride, padding=1, groups=cur_cout, bias=False),
                nn.BatchNorm2d(cur_cout),
                nn.ReLU6(inplace=True),
                # pw-linear
                #*
                # nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                FuseConv2d(last_cout2, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(cur_cout2),
            )
            
            last_cout = last_cout2
            cur_cout = cur_cout2

        else:
            last_cout = cur_cout
            cur_cout = int(hidden_dim * (1-layers_cprate[cur_convid]))
            cur_convid += 1

            last_cout2 = cur_cout
            cur_cout2 = last_cout2

            last_cout3 = cur_cout2
            cur_cout3 = int(oup * (1-layers_cprate[cur_convid]))
            cur_convid += 1
            self.conv = nn.Sequential(
                # pw
                #*
                # nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                FuseConv2d(last_cout, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(cur_cout),
                nn.ReLU6(inplace=True),
                # dw
                
                nn.Conv2d(last_cout2, cur_cout2, kernel_size=3, stride=stride, padding=1, groups=cur_cout2, bias=False),
                nn.BatchNorm2d(cur_cout2),
                nn.ReLU6(inplace=True),
                # pw-linear
                #*
                # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                FuseConv2d(last_cout3, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(cur_cout3),
            )
            last_cout = last_cout3
            cur_cout = cur_cout3

        if self.use_res_connect:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        if self.use_res_connect:
            out = self.conv(x)
            out += self.shortcut(x)
            return out
        else:
            return self.conv(x)


class FusedMobileNetV2(nn.Module):
    def __init__(self, layers_cprate, n_class=1000, input_size=224, width_mult=1.0):

        global cur_convid, last_cout, cur_cout

        super(FusedMobileNetV2, self).__init__()
        block = FusedInvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        #*
        self.features = [fused_conv_bn(3, input_channel, 2, layers_cprate)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = fused_make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, layers_cprate=layers_cprate))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, layers_cprate=layers_cprate))
                input_channel = output_channel
        #*
        self.last_channel = fused_make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        # building last several layers
        self.features.append(fused_conv_1x1_bn(input_channel, self.last_channel, layers_cprate))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        last_cout = cur_cout
        self.classifier = nn.Linear(last_cout, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def fused_mobilenetv2(layers_cprate):
    global cur_convid, last_cout, cur_cout
    cur_convid = 0
    last_cout = -1
    cur_cout = 3
    return FusedMobileNetV2(layers_cprate = layers_cprate, width_mult=1.0)




#*================================
#* Compact model
def compact_conv_bn(inp, oup, stride, layers_cprate):
    global cur_convid, last_cout, cur_cout

    last_cout = cur_cout
    cur_cout = int(oup * (1-layers_cprate[cur_convid]))
    cur_convid += 1
    return nn.Sequential(
        nn.Conv2d(last_cout, cur_cout, kernel_size=3, stride=stride, padding=1, bias=False),
        # FuseConv2d(last_cout, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cur_cout),
        nn.ReLU6(inplace=True)
    )

def compact_conv_1x1_bn(inp, oup, layers_cprate):
    global cur_convid, last_cout, cur_cout
    #*
    last_cout = cur_cout
    cur_cout = int(oup * (1-layers_cprate[cur_convid]))
    cur_convid += 1
    return nn.Sequential(
        nn.Conv2d(last_cout, cur_cout, kernel_size=1, stride=1, padding=0, bias=False),
        # FuseConv2d(last_cout, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(cur_cout),
        nn.ReLU6(inplace=True)
    )

def compact_make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class CompactInvertedResidual(nn.Module):
    
    def __init__(self, inp, oup, stride, expand_ratio, layers_cprate):
        global cur_convid, last_cout, cur_cout
        super(CompactInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            last_cout = cur_cout
            cur_cout = last_cout

            last_cout2 = cur_cout
            cur_cout2 = int(oup * (1-layers_cprate[cur_convid]))
            cur_convid += 1
            self.conv = nn.Sequential(
                # dw
                # nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.Conv2d(last_cout, cur_cout, kernel_size=3, stride=stride, padding=1, groups=cur_cout, bias=False),
                nn.BatchNorm2d(cur_cout),
                nn.ReLU6(inplace=True),
                # pw-linear
                #*
                nn.Conv2d(last_cout2, cur_cout2, kernel_size=1, stride=1, padding=0, bias=False),
                # FuseConv2d(last_cout2, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(cur_cout2),
            )
            
            last_cout = last_cout2
            cur_cout = cur_cout2

        else:
            last_cout = cur_cout
            cur_cout = int(hidden_dim * (1-layers_cprate[cur_convid]))
            cur_convid += 1

            last_cout2 = cur_cout
            cur_cout2 = last_cout2

            last_cout3 = cur_cout2
            cur_cout3 = int(oup * (1-layers_cprate[cur_convid]))
            cur_convid += 1

            self.conv = nn.Sequential(
                # pw
                #*
                nn.Conv2d(last_cout, cur_cout, kernel_size=1, stride=1, padding=0, bias=False),
                # FuseConv2d(last_cout, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(cur_cout),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(last_cout2, cur_cout2, kernel_size=3, stride=stride, padding=1, groups=cur_cout2, bias=False),
                nn.BatchNorm2d(cur_cout2),
                nn.ReLU6(inplace=True),
                # pw-linear
                #*
                nn.Conv2d(last_cout3, cur_cout3, 1, 1, 0, bias=False),
                # FuseConv2d(last_cout3, cur_cout3, 1, 1, 0, bias=False),
                nn.BatchNorm2d(cur_cout3),
            )
            last_cout = last_cout3
            cur_cout = cur_cout3

        if self.use_res_connect:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        if self.use_res_connect:
            out = self.conv(x)
            out += self.shortcut(x)
            return out
        else:
            return self.conv(x)


class CompactMobileNetV2(nn.Module):
    def __init__(self, layers_cprate, n_class=1000, input_size=224, width_mult=1.0):

        global cur_convid, last_cout, cur_cout

        super(CompactMobileNetV2, self).__init__()
        block = CompactInvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        #*
        self.features = [compact_conv_bn(3, input_channel, 2, layers_cprate)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = compact_make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, layers_cprate=layers_cprate))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, layers_cprate=layers_cprate))
                input_channel = output_channel
        #*
        self.last_channel = compact_make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        # building last several layers
        self.features.append(compact_conv_1x1_bn(input_channel, self.last_channel, layers_cprate))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        last_cout = cur_cout
        self.classifier = nn.Linear(last_cout, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def compact_mobilenetv2(layers_cprate):
    global cur_convid, last_cout, cur_cout
    cur_convid = 0
    last_cout = -1
    cur_cout = 3
    return CompactMobileNetV2(layers_cprate = layers_cprate, width_mult=1.0)





#*================================
#* Origin model
def origin_conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def origin_conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def origin_make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class OriginInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(OriginInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        if self.use_res_connect:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        if self.use_res_connect:
            out = self.conv(x)
            out += self.shortcut(x)
            return out
        else:
            return self.conv(x)


class OriginMobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(OriginMobileNetV2, self).__init__()
        block = OriginInvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = origin_make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = origin_make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [origin_conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = origin_make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(origin_conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def origin_mobilenetv2():
    return OriginMobileNetV2(width_mult=1)








# def test():
#     model  = compact_mobilenetv2(cprate)
#     print(model)
#     x = torch.randn(1, 3, 224, 224)
#     y = model(x)
#     print(y.shape)



# test()

