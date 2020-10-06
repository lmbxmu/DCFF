'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

#* 
from .fuse_modules import FuseConv2d
# from fuse_modules import FuseConv2d

#*====================================================
#* Fused model
class FusedInception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, flayer_cprate, cur_fconvid):
        super(FusedInception, self).__init__()
        #* 1x1 conv branch
        cur_fconvid += 0
        last_cout = in_planes
        cur_cout = int(n1x1*(1-flayer_cprate[cur_fconvid]))

        self.b1 = nn.Sequential(
            # nn.Conv2d(in_planes, n1x1, kernel_size=1),
            FuseConv2d(last_cout, n1x1, kernel_size=1),
            nn.BatchNorm2d(cur_cout),
            nn.ReLU(True),
        )

        #* 1x1 conv -> 3x3 conv branch
        cur_fconvid += 1
        last_cout0 = in_planes
        cur_cout0 = int(n3x3red*(1-flayer_cprate[cur_fconvid]))

        cur_fconvid += 1
        last_cout1 = cur_cout0
        cur_cout1 = int(n3x3*(1-flayer_cprate[cur_fconvid]))

        self.b2 = nn.Sequential(
            # nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            FuseConv2d(last_cout0, n3x3red, kernel_size=1),
            nn.BatchNorm2d(cur_cout0),
            nn.ReLU(True),
            # nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            FuseConv2d(last_cout1, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout1),
            nn.ReLU(True),
        )
        cur_cout = cur_cout1

        #* 1x1 conv -> 5x5 conv branch
        cur_fconvid += 1
        last_cout0 = in_planes
        cur_cout0 = int(n5x5red*(1-flayer_cprate[cur_fconvid]))

        cur_fconvid += 1
        last_cout1 = cur_cout0
        cur_cout1 = int(n5x5*(1-flayer_cprate[cur_fconvid]))

        cur_fconvid += 1
        last_cout2 = cur_cout1
        cur_cout2 = int(n5x5*(1-flayer_cprate[cur_fconvid]))

        self.b3 = nn.Sequential(
            # nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            FuseConv2d(last_cout0, n5x5red, kernel_size=1),
            nn.BatchNorm2d(cur_cout0),
            nn.ReLU(True),
            # nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            FuseConv2d(last_cout1, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout1),
            nn.ReLU(True),
            # nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            FuseConv2d(last_cout2, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout2),
            nn.ReLU(True),
        )
        cur_cout = cur_cout2

        # 3x3 pool -> 1x1 conv branch
        cur_fconvid += 1
        last_cout = in_planes
        cur_cout = int(pool_planes*(1-flayer_cprate[cur_fconvid]))

        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            # nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            FuseConv2d(last_cout, pool_planes, kernel_size=1),
            nn.BatchNorm2d(cur_cout),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class FusedGoogLeNet(nn.Module):
    def __init__(self, flayer_cprate):
        super(FusedGoogLeNet, self).__init__()
        #*
        cur_fconvid = 0
        last_stage_cout = 3
        cur_stage_cout = int(192*(1-flayer_cprate[cur_fconvid]))
        self.pre_layers = nn.Sequential(
            # nn.Conv2d(3, 192, kernel_size=3, padding=1),
            FuseConv2d(last_stage_cout, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_stage_cout),
            nn.ReLU(True),
        )

        #*
        cur_fconvid += 1
        last_stage_cout = cur_stage_cout
        b1_cout = int( 64*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(128*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int( 32*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int( 32*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.a3 = FusedInception(last_stage_cout,     64,     96, 128,    16,  32,     32,    flayer_cprate, cur_fconvid)

        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(128*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(192*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int( 96*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int( 64*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        
        self.b3 = FusedInception(last_stage_cout,    128,    128, 192,    32,  96,     64,    flayer_cprate, cur_fconvid)


        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)


        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(192*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(208*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int( 48*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int( 64*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.a4 = FusedInception(last_stage_cout,    192,     96, 208,    16,  48,     64,    flayer_cprate, cur_fconvid)

        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(160*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(224*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int( 64*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int( 64*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.b4 = FusedInception(last_stage_cout,    160,    112, 224,    24,  64,     64,    flayer_cprate, cur_fconvid)

        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(128*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(256*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int( 64*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int( 64*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.c4 = FusedInception(last_stage_cout,    128,    128, 256,    24,  64,     64,    flayer_cprate, cur_fconvid)

        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(112*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(288*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int( 64*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int( 64*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.d4 = FusedInception(last_stage_cout,    112,    144, 288,    32,  64,     64,    flayer_cprate, cur_fconvid)

        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(256*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(320*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int(128*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int(128*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.e4 = FusedInception(last_stage_cout,    256,    160, 320,    32, 128,    128,    flayer_cprate, cur_fconvid)


        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(256*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(320*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int(128*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int(128*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.a5 = FusedInception(last_stage_cout,    256,    160, 320,    32, 128,    128,    flayer_cprate, cur_fconvid)

        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(384*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(384*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int(128*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int(128*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.b5 = FusedInception(last_stage_cout,    384,    192, 384,    48, 128,    128,    flayer_cprate, cur_fconvid)


        self.avgpool = nn.AvgPool2d(8, stride=1)

        last_stage_cout = cur_stage_cout
        self.linear = nn.Linear(last_stage_cout, 10)

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
#* Compact model
class CompactInception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, flayer_cprate, cur_fconvid):
        super(CompactInception, self).__init__()
        #* 1x1 conv branch
        cur_fconvid += 0
        last_cout = in_planes
        cur_cout = int(n1x1*(1-flayer_cprate[cur_fconvid]))

        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, cur_cout, kernel_size=1),
            # FuseConv2d(last_cout, n1x1, kernel_size=1),
            nn.BatchNorm2d(cur_cout),
            nn.ReLU(True),
        )

        #* 1x1 conv -> 3x3 conv branch
        cur_fconvid += 1
        last_cout0 = in_planes
        cur_cout0 = int(n3x3red*(1-flayer_cprate[cur_fconvid]))

        cur_fconvid += 1
        last_cout1 = cur_cout0
        cur_cout1 = int(n3x3*(1-flayer_cprate[cur_fconvid]))

        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, cur_cout0, kernel_size=1),
            # FuseConv2d(last_cout0, n3x3red, kernel_size=1),
            nn.BatchNorm2d(cur_cout0),
            nn.ReLU(True),
            nn.Conv2d(cur_cout0, cur_cout1, kernel_size=3, padding=1),
            # FuseConv2d(last_cout1, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout1),
            nn.ReLU(True),
        )
        cur_cout = cur_cout1

        #* 1x1 conv -> 5x5 conv branch
        cur_fconvid += 1
        last_cout0 = in_planes
        cur_cout0 = int(n5x5red*(1-flayer_cprate[cur_fconvid]))

        cur_fconvid += 1
        last_cout1 = cur_cout0
        cur_cout1 = int(n5x5*(1-flayer_cprate[cur_fconvid]))

        cur_fconvid += 1
        last_cout2 = cur_cout1
        cur_cout2 = int(n5x5*(1-flayer_cprate[cur_fconvid]))

        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, cur_cout0, kernel_size=1),
            # FuseConv2d(last_cout0, n5x5red, kernel_size=1),
            nn.BatchNorm2d(cur_cout0),
            nn.ReLU(True),
            nn.Conv2d(cur_cout0, cur_cout1, kernel_size=3, padding=1),
            # FuseConv2d(last_cout1, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout1),
            nn.ReLU(True),
            nn.Conv2d(cur_cout1, cur_cout2, kernel_size=3, padding=1),
            # FuseConv2d(last_cout2, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_cout2),
            nn.ReLU(True),
        )
        cur_cout = cur_cout2

        # 3x3 pool -> 1x1 conv branch
        cur_fconvid += 1
        last_cout = in_planes
        cur_cout = int(pool_planes*(1-flayer_cprate[cur_fconvid]))

        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(last_cout, cur_cout, kernel_size=1),
            # FuseConv2d(last_cout, pool_planes, kernel_size=1),
            nn.BatchNorm2d(cur_cout),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class CompactGoogLeNet(nn.Module):
    def __init__(self, flayer_cprate):
        super(CompactGoogLeNet, self).__init__()
        #*
        cur_fconvid = 0
        last_stage_cout = 3
        cur_stage_cout = int(192*(1-flayer_cprate[cur_fconvid]))
        self.pre_layers = nn.Sequential(
            nn.Conv2d(last_stage_cout, cur_stage_cout, kernel_size=3, padding=1),
            # FuseConv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_stage_cout),
            nn.ReLU(True),
        )

        #*
        cur_fconvid += 1
        last_stage_cout = cur_stage_cout
        b1_cout = int( 64*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(128*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int( 32*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int( 32*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.a3 = CompactInception(last_stage_cout,     64,     96, 128,    16,  32,     32,    flayer_cprate, cur_fconvid)

        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(128*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(192*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int( 96*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int( 64*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        
        self.b3 = CompactInception(last_stage_cout,    128,    128, 192,    32,  96,     64,    flayer_cprate, cur_fconvid)


        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)


        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(192*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(208*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int( 48*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int( 64*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.a4 = CompactInception(last_stage_cout,    192,     96, 208,    16,  48,     64,    flayer_cprate, cur_fconvid)

        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(160*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(224*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int( 64*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int( 64*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.b4 = CompactInception(last_stage_cout,    160,    112, 224,    24,  64,     64,    flayer_cprate, cur_fconvid)

        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(128*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(256*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int( 64*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int( 64*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.c4 = CompactInception(last_stage_cout,    128,    128, 256,    24,  64,     64,    flayer_cprate, cur_fconvid)

        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(112*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(288*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int( 64*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int( 64*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.d4 = CompactInception(last_stage_cout,    112,    144, 288,    32,  64,     64,    flayer_cprate, cur_fconvid)

        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(256*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(320*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int(128*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int(128*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.e4 = CompactInception(last_stage_cout,    256,    160, 320,    32, 128,    128,    flayer_cprate, cur_fconvid)


        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(256*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(320*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int(128*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int(128*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.a5 = CompactInception(last_stage_cout,    256,    160, 320,    32, 128,    128,    flayer_cprate, cur_fconvid)

        cur_fconvid += 7
        last_stage_cout = cur_stage_cout
        b1_cout = int(384*(1-flayer_cprate[cur_fconvid]))
        b2_cout = int(384*(1-flayer_cprate[cur_fconvid+2]))
        b3_cout = int(128*(1-flayer_cprate[cur_fconvid+5]))
        b4_cout = int(128*(1-flayer_cprate[cur_fconvid+6]))
        cur_stage_cout = b1_cout + b2_cout + b3_cout + b4_cout
        self.b5 = CompactInception(last_stage_cout,    384,    192, 384,    48, 128,    128,    flayer_cprate, cur_fconvid)


        self.avgpool = nn.AvgPool2d(8, stride=1)

        last_stage_cout = cur_stage_cout
        self.linear = nn.Linear(last_stage_cout, 10)

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
    flayer_cprate = [0.9]*64
    model = CompactGoogLeNet(flayer_cprate)
    # model = OriginGoogLeNet()
    print(model)

    x = torch.randn(1,3,32,32)
    y = model(x)
    print(y.size())

# test()