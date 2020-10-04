import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

#* 
from .fuse_modules import FuseConv2d
# from fuse_modules import FuseConv2d


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

#*====================================================
#* fused model
def fused_conv3x3(in_planes, out_planes, stride=1):
    return FuseConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FusedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, cprate, cur_fconvid, stride=1):
        super(FusedBasicBlock, self).__init__()

        #*
        real_in_planes = int(in_planes*(1-cprate[cur_fconvid-1]))
        real_planes = int(planes*(1-cprate[cur_fconvid]))

        last_cout = real_in_planes
        cur_cout = real_planes

        self.conv1 = fused_conv3x3(last_cout, planes, stride)
        self.bn1 = nn.BatchNorm2d(cur_cout)
        self.relu = nn.ReLU(inplace=True)

        cur_fconvid = cur_fconvid + 1

        
        last_cout = int(planes*(1-cprate[cur_fconvid-1]))
        cur_cout = int(planes*(1-cprate[cur_fconvid]))

        self.conv2 = fused_conv3x3(last_cout, planes)
        self.bn2 = nn.BatchNorm2d(cur_cout)

        self.shortcut = nn.Sequential()

        cur_cout = int(planes*(1-cprate[cur_fconvid]))

        
        if stride != 1 or real_in_planes != cur_cout:
            if stride!=1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (cur_cout-real_in_planes)//2, cur_cout-real_in_planes-(cur_cout-real_in_planes)//2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (cur_cout-real_in_planes)//2, cur_cout-real_in_planes-(cur_cout-real_in_planes)//2), "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class FusedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cprate=None):
        super(FusedResNet, self).__init__()
        #*
        self.cprate = cprate
        self.cur_fconvid = 0

        self.in_planes = 16

        #*
        self.conv1 = fused_conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(int(16*(1-self.cprate[self.cur_fconvid])))
        self.relu = nn.ReLU(inplace=True)
        #*
        self.cur_fconvid = self.cur_fconvid +1

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        cout = int(64*(1-self.cprate[-1]))

        self.fc = nn.Linear(cout, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, 
                                cprate = self.cprate, cur_fconvid=self.cur_fconvid, stride=stride))
            self.in_planes = planes * block.expansion
            #*
            self.cur_fconvid = self.cur_fconvid + 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def fused_resnet56(cprate):
    return FusedResNet(FusedBasicBlock, [9, 9, 9], cprate=cprate)

def fused_resnet110(cprate):
    return FusedResNet(FusedBasicBlock, [18, 18, 18], cprate=cprate)



#*====================================================
#* compact model
def compact_conv3x3(in_planes, out_planes, stride=1):
    # return FuseConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



class CompactBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, cprate, cur_fconvid, stride=1):
        super(CompactBasicBlock, self).__init__()

        #*
        real_in_planes = int(in_planes*(1-cprate[cur_fconvid-1]))
        real_planes = int(planes*(1-cprate[cur_fconvid]))

        last_cout = real_in_planes
        cur_cout = real_planes

        # self.conv1 = compact_conv3x3(last_cout, planes, stride)
        self.conv1 = compact_conv3x3(last_cout, cur_cout, stride)
        self.bn1 = nn.BatchNorm2d(cur_cout)
        self.relu = nn.ReLU(inplace=True)

        cur_fconvid = cur_fconvid + 1

        
        last_cout = int(planes*(1-cprate[cur_fconvid-1]))
        cur_cout = int(planes*(1-cprate[cur_fconvid]))

        # self.conv2 = compact_conv3x3(last_cout, planes)
        self.conv2 = compact_conv3x3(last_cout, cur_cout)
        self.bn2 = nn.BatchNorm2d(cur_cout)

        self.shortcut = nn.Sequential()

        cur_cout = int(planes*(1-cprate[cur_fconvid]))

        
        if stride != 1 or real_in_planes != cur_cout:
            if stride!=1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (cur_cout-real_in_planes)//2, cur_cout-real_in_planes-(cur_cout-real_in_planes)//2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (cur_cout-real_in_planes)//2, cur_cout-real_in_planes-(cur_cout-real_in_planes)//2), "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class CompactResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cprate=None):
        super(CompactResNet, self).__init__()
        #*
        self.cprate = cprate
        self.cur_fconvid = 0

        self.in_planes = 16

        #*
        cur_cout = int(16*(1-self.cprate[self.cur_fconvid]))
        self.conv1 = compact_conv3x3(3, cur_cout)
        self.bn1 = nn.BatchNorm2d(cur_cout)
        self.relu = nn.ReLU(inplace=True)
        #*
        self.cur_fconvid = self.cur_fconvid +1

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        cout = int(64*(1-self.cprate[-1]))

        self.fc = nn.Linear(cout, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, planes=planes, 
                                cprate = self.cprate, cur_fconvid=self.cur_fconvid, stride=stride))
            self.in_planes = planes * block.expansion
            #*
            self.cur_fconvid = self.cur_fconvid + 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def compact_resnet56(cprate):
    return CompactResNet(CompactBasicBlock, [9, 9, 9], cprate=cprate)

def compact_resnet110(cprate):
    return CompactResNet(CompactBasicBlock, [18, 18, 18], cprate=cprate)





#*====================================================
#* origin model
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class OriginBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(OriginBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

        # if stride != 1 or in_planes != planes:
        #     if stride!=1:
        #         self.shortcut = LambdaLayer(
        #             lambda x: F.pad(x[:, :, ::2, ::2],
        #                             (0, 0, 0, 0, (planes-in_planes)//2, planes-in_planes-(planes-in_planes)//2), "constant", 0))
        #     else:
        #         self.shortcut = LambdaLayer(
        #             lambda x: F.pad(x[:, :, :, :],
        #                             (0, 0, 0, 0, (planes-in_planes)//2, planes-in_planes-(planes-in_planes)//2), "constant", 0))


    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class OriginResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(OriginResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        
        out = self.layer2(out)

        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def origin_resnet56():
    return OriginResNet(OriginBasicBlock, [9, 9, 9])


def origin_resnet110():
    return OriginResNet(OriginBasicBlock, [18, 18, 18])






# def test():
#     cprate = [0.0]*30
#     # model = compact_resnet56(cprate)
#     model = origin_resnet56()
#     print(model)
#     # x = torch.randn(2,3,32,32)
#     # y = model(x)
#     # print(y.size())

# test()
