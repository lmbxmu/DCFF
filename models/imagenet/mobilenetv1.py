import torch.nn as nn
import torch

from .fuse_modules import FuseConv2d
# from fuse_modules import FuseConv2d

#*================================
#* Fused model
class FusedMobileNetV1(nn.Module):
    def __init__(self, cprate, num_classes=1000):
        super(FusedMobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride, last_cprate, cur_cprate):
            last_cout = int(inp*(1-last_cprate))
            cur_cout = int(oup*(1-cur_cprate))
            return nn.Sequential(
                # nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                FuseConv2d(last_cout, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(cur_cout),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride, last_cprate, cur_cprate):
            last_cout = int(inp*(1-last_cprate))
            cur_cout = int(oup*(1-cur_cprate))
            return nn.Sequential(
                #* dw
                nn.Conv2d(last_cout, last_cout, 3, stride, 1, groups=last_cout, bias=False),
                # FuseConv2d(last_cout, last_cout, 3, stride, 1, groups=last_cout, bias=False),
                nn.BatchNorm2d(last_cout),
                nn.ReLU(inplace=True),

                #* pw
                # nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                FuseConv2d(last_cout, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(cur_cout),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            
            conv_bn(   3,   32, 2,       0.0,  cprate[0]),

            conv_dw(  32,   64, 1, cprate[0],  cprate[1]),
            conv_dw(  64,  128, 2, cprate[1],  cprate[2]),
            conv_dw( 128,  128, 1, cprate[2],  cprate[3]),
            conv_dw( 128,  256, 2, cprate[3],  cprate[4]),
            conv_dw( 256,  256, 1, cprate[4],  cprate[5]),
            conv_dw( 256,  512, 2, cprate[5],  cprate[6]),
            conv_dw( 512,  512, 1, cprate[6],  cprate[7]),
            conv_dw( 512,  512, 1, cprate[7],  cprate[8]),
            conv_dw( 512,  512, 1, cprate[8],  cprate[9]),
            conv_dw( 512,  512, 1, cprate[9],  cprate[10]),
            conv_dw( 512,  512, 1, cprate[10], cprate[11]),
            conv_dw( 512, 1024, 2, cprate[11], cprate[12]),
            conv_dw(1024, 1024, 1, cprate[12], cprate[13]),

            nn.AdaptiveAvgPool2d(1)
        )
        last_cout = int(1024*(1-cprate[13]))
        self.fc = nn.Linear(last_cout, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def fused_mobilenetv1(cprate):
    return FusedMobileNetV1(cprate)





#*================================
#* Compact model
class CompactMobileNetV1(nn.Module):
    def __init__(self, cprate, num_classes=1000):
        super(CompactMobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride, last_cprate, cur_cprate):
            last_cout = int(inp*(1-last_cprate))
            cur_cout = int(oup*(1-cur_cprate))
            return nn.Sequential(
                nn.Conv2d(last_cout, cur_cout, 3, stride, 1, bias=False),
                # FuseConv2d(last_cout, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(cur_cout),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride, last_cprate, cur_cprate):
            last_cout = int(inp*(1-last_cprate))
            cur_cout = int(oup*(1-cur_cprate))
            return nn.Sequential(
                #* dw
                nn.Conv2d(last_cout, last_cout, 3, stride, 1, groups=last_cout, bias=False),
                # FuseConv2d(last_cout, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(last_cout),
                nn.ReLU(inplace=True),

                #* pw
                nn.Conv2d(last_cout, cur_cout, 1, 1, 0, bias=False),
                # FuseConv2d(last_cout, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(cur_cout),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            
            conv_bn(   3,   32, 2,       0.0,  cprate[0]),

            conv_dw(  32,   64, 1, cprate[0],  cprate[1]),
            conv_dw(  64,  128, 2, cprate[1],  cprate[2]),
            conv_dw( 128,  128, 1, cprate[2],  cprate[3]),
            conv_dw( 128,  256, 2, cprate[3],  cprate[4]),
            conv_dw( 256,  256, 1, cprate[4],  cprate[5]),
            conv_dw( 256,  512, 2, cprate[5],  cprate[6]),
            conv_dw( 512,  512, 1, cprate[6],  cprate[7]),
            conv_dw( 512,  512, 1, cprate[7],  cprate[8]),
            conv_dw( 512,  512, 1, cprate[8],  cprate[9]),
            conv_dw( 512,  512, 1, cprate[9],  cprate[10]),
            conv_dw( 512,  512, 1, cprate[10], cprate[11]),
            conv_dw( 512, 1024, 2, cprate[11], cprate[12]),
            conv_dw(1024, 1024, 1, cprate[12], cprate[13]),

            nn.AdaptiveAvgPool2d(1)
        )
        last_cout = int(1024*(1-cprate[13]))
        self.fc = nn.Linear(last_cout, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def compact_mobilenetv1(cprate):
    return CompactMobileNetV1(cprate)







#*================================
#* Origin model
class OriginMobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(OriginMobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),

            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),

            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def origin_mobilenetv1():
    return OriginMobileNetV1()

# def test():
#     cprate = [0.9]*14
#     # model = origin_mobilenetv1()
#     model = compact_mobilenetv1(cprate)
#     print(model)
#     x = torch.randn(1, 3, 224, 224)
#     y = model(x)
#     print(y.shape)

# test()
