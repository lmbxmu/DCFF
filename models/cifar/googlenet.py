'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

#* 
from .fuse_modules import FuseConv2d
# from fuse_modules import FuseConv2d

#*====================================================
#* fused model
class FusedInception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, cprate, cur_stageid):
        super(FusedInception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        #*
        cur_cout = int(n3x3*(1-cprate[cur_stageid]))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            # nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            FuseConv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        #*
        cur_cout = int(n5x5*(1-cprate[cur_stageid]))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            # nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            FuseConv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout),
            nn.ReLU(True),
            # nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            FuseConv2d(cur_cout, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        # print(x.shape)
        # print(y1.shape)
        # exit(0)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class FusedGoogLeNet(nn.Module):
    def __init__(self, cprate):
        super(FusedGoogLeNet, self).__init__()
        #*
        print(cprate)
        # exit(0)
        cur_stageid = 0
        cur_cout = int(192*(1-cprate[cur_stageid]))
        self.pre_layers = nn.Sequential(
            # nn.Conv2d(3, 192, kernel_size=3, padding=1),
            FuseConv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout),
            nn.ReLU(True),
        )

        #*
        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 64 + int(128*cur_hold_rate) + int(32*cur_hold_rate) + 32
        print(cprate, cur_stageid)
        # exit(0)
        self.a3 = FusedInception(last_cout,     64,     96, 128,    16,  32,     32,    cprate, cur_stageid)

        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 128 + int(192*cur_hold_rate) + int(96*cur_hold_rate) + 64
        self.b3 = FusedInception(last_cout,    128,    128, 192,    32,  96,     64,    cprate, cur_stageid)


        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)


        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 192 + int(208*cur_hold_rate) + int(48*cur_hold_rate) + 64
        self.a4 = FusedInception(last_cout,    192,     96, 208,    16,  48,     64,    cprate, cur_stageid)

        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 160 + int(224*cur_hold_rate) + int(64*cur_hold_rate) + 64
        self.b4 = FusedInception(last_cout,    160,    112, 224,    24,  64,     64,    cprate, cur_stageid)

        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 128 + int(256*cur_hold_rate) + int(64*cur_hold_rate) + 64
        self.c4 = FusedInception(last_cout,    128,    128, 256,    24,  64,     64,    cprate, cur_stageid)

        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 112 + int(288*cur_hold_rate) + int(64*cur_hold_rate) + 64
        self.d4 = FusedInception(last_cout,    112,    144, 288,    32,  64,     64,    cprate, cur_stageid)

        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 256 + int(320*cur_hold_rate) + int(128*cur_hold_rate) + 128
        self.e4 = FusedInception(last_cout,    256,    160, 320,    32, 128,    128,    cprate, cur_stageid)


        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 256 + int(320*cur_hold_rate) + int(128*cur_hold_rate) + 128
        self.a5 = FusedInception(last_cout,    256,    160, 320,    32, 128,    128,    cprate, cur_stageid)

        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 384 + int(384*cur_hold_rate) + int(128*cur_hold_rate) + 128
        # print(cur_stageid, cur_hold_rate)
        # exit(0)
        self.b5 = FusedInception(last_cout,    384,    192, 384,    48, 128,    128,    cprate, cur_stageid)


        self.avgpool = nn.AvgPool2d(8, stride=1)

        cur_stageid += 1
        last_cout = cur_cout
        self.linear = nn.Linear(last_cout, 10)

    def forward(self, x):
        # print(x.shape)
        # exit(0)
        out = self.pre_layers(x)

        # 192 x 32 x 32
        out = self.a3(out)
        # 256 x 32 x 32
        out = self.b3(out)
        # 480 x 32 x 32
        out = self.maxpool(out)

        # 480 x 16 x 16
        out = self.a4(out)
        # 512 x 16 x 16
        out = self.b4(out)
        # 512 x 16 x 16
        out = self.c4(out)
        # 512 x 16 x 16
        out = self.d4(out)
        # 528 x 16 x 16
        out = self.e4(out)
        # 823 x 16 x 16
        out = self.maxpool(out)

        # 823 x 8 x 8
        out = self.a5(out)
        # 823 x 8 x 8
        out = self.b5(out)

        # 1024 x 8 x 8
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out





#*====================================================
#* fused model
class CompactInception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, cprate, cur_stageid):
        super(CompactInception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        #*
        cur_cout = int(n3x3*(1-cprate[cur_stageid]))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, cur_cout, kernel_size=3, padding=1),
            # FuseConv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        #*
        cur_cout = int(n5x5*(1-cprate[cur_stageid]))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, cur_cout, kernel_size=3, padding=1),
            # FuseConv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout),
            nn.ReLU(True),
            nn.Conv2d(cur_cout, cur_cout, kernel_size=3, padding=1),
            # FuseConv2d(cur_cout, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class CompactGoogLeNet(nn.Module):
    def __init__(self, cprate):
        super(CompactGoogLeNet, self).__init__()
        #*
        cur_stageid = 0
        cur_cout = int(192*(1-cprate[cur_stageid]))
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, cur_cout, kernel_size=3, padding=1),
            # FuseConv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout),
            nn.ReLU(True),
        )

        #*
        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 64 + int(128*cur_hold_rate) + int(32*cur_hold_rate) + 32
        self.a3 = CompactInception(last_cout,     64,     96, 128,    16,  32,     32,    cprate, cur_stageid)

        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 128 + int(192*cur_hold_rate) + int(96*cur_hold_rate) + 64
        self.b3 = CompactInception(last_cout,    128,    128, 192,    32,  96,     64,    cprate, cur_stageid)


        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)


        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 192 + int(208*cur_hold_rate) + int(48*cur_hold_rate) + 64
        self.a4 = CompactInception(last_cout,    192,     96, 208,    16,  48,     64,    cprate, cur_stageid)

        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 160 + int(224*cur_hold_rate) + int(64*cur_hold_rate) + 64
        self.b4 = CompactInception(last_cout,    160,    112, 224,    24,  64,     64,    cprate, cur_stageid)

        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 128 + int(256*cur_hold_rate) + int(64*cur_hold_rate) + 64
        self.c4 = CompactInception(last_cout,    128,    128, 256,    24,  64,     64,    cprate, cur_stageid)

        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 112 + int(288*cur_hold_rate) + int(64*cur_hold_rate) + 64
        self.d4 = CompactInception(last_cout,    112,    144, 288,    32,  64,     64,    cprate, cur_stageid)

        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 256 + int(320*cur_hold_rate) + int(128*cur_hold_rate) + 128
        self.e4 = CompactInception(last_cout,    256,    160, 320,    32, 128,    128,    cprate, cur_stageid)


        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 256 + int(320*cur_hold_rate) + int(128*cur_hold_rate) + 128
        self.a5 = CompactInception(last_cout,    256,    160, 320,    32, 128,    128,    cprate, cur_stageid)

        cur_stageid += 1
        cur_hold_rate = (1-cprate[cur_stageid])
        last_cout = cur_cout
        cur_cout = 384 + int(384*cur_hold_rate) + int(128*cur_hold_rate) + 128

        self.b5 = CompactInception(last_cout,    384,    192, 384,    48, 128,    128,    cprate, cur_stageid)


        self.avgpool = nn.AvgPool2d(8, stride=1)

        cur_stageid += 1
        last_cout = cur_cout
        self.linear = nn.Linear(last_cout, 10)

    def forward(self, x):
        out = self.pre_layers(x)

        # 192 x 32 x 32
        out = self.a3(out)
        # 256 x 32 x 32
        out = self.b3(out)
        # 480 x 32 x 32
        out = self.maxpool(out)

        # 480 x 16 x 16
        out = self.a4(out)
        # 512 x 16 x 16
        out = self.b4(out)
        # 512 x 16 x 16
        out = self.c4(out)
        # 512 x 16 x 16
        out = self.d4(out)
        # 528 x 16 x 16
        out = self.e4(out)
        # 823 x 16 x 16
        out = self.maxpool(out)

        # 823 x 8 x 8
        out = self.a5(out)
        # 823 x 8 x 8
        out = self.b5(out)

        # 1024 x 8 x 8
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out





#*====================================================
#* origin model
class OriginInception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(OriginInception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class OriginGoogLeNet(nn.Module):
    def __init__(self):
        super(OriginGoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = OriginInception(192,    64,      96, 128,    16,  32,     32)
        self.b3 = OriginInception(256,    128,    128, 192,    32,  96,     64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = OriginInception(480,    192,     96, 208,    16,  48,     64)
        self.b4 = OriginInception(512,    160,    112, 224,    24,  64,     64)
        self.c4 = OriginInception(512,    128,    128, 256,    24,  64,     64)
        self.d4 = OriginInception(512,    112,    144, 288,    32,  64,     64)
        self.e4 = OriginInception(528,    256,    160, 320,    32, 128,    128)

        self.a5 = OriginInception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = OriginInception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    cprate = [0.5]*10
    model = CompactGoogLeNet(cprate)
    # model = OriginGoogLeNet()
    print(model)

    x = torch.randn(1,3,32,32)
    y = model(x)
    print(y.size())

# test()
