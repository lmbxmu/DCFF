import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

#* 
from .fuse_modules import FuseConv2d

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)



def conv3x3(in_planes, out_planes, stride=1):
    return FuseConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, cprate, cur_convid, stride=1):
        super(BasicBlock, self).__init__()

        #*
        real_in_planes = int(in_planes*(1-cprate[cur_convid-1]))
        real_planes = int(planes*(1-cprate[cur_convid]))

        last_cout = real_in_planes
        cur_cout = real_planes

        self.conv1 = conv3x3(last_cout, planes, stride)
        self.bn1 = nn.BatchNorm2d(cur_cout)
        self.relu = nn.ReLU(inplace=True)

        cur_convid = cur_convid + 1

        
        last_cout = int(planes*(1-cprate[cur_convid-1]))
        cur_cout = int(planes*(1-cprate[cur_convid]))

        self.conv2 = conv3x3(last_cout, planes)
        self.bn2 = nn.BatchNorm2d(cur_cout)

        self.shortcut = nn.Sequential()

        cur_cout = int(planes*(1-cprate[cur_convid]))

        
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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cprate=None):
        super(ResNet, self).__init__()
        #*
        self.cprate = cprate
        self.cur_convid = 0

        self.in_planes = 16

        #*
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(int(16*(1-self.cprate[self.cur_convid])))
        self.relu = nn.ReLU(inplace=True)
        #*
        self.cur_convid = self.cur_convid +1

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
                                cprate = self.cprate, cur_convid=self.cur_convid, stride=stride))
            self.in_planes = planes * block.expansion
            #*
            self.cur_convid = self.cur_convid + 2
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


def resnet20(cprate):
    return ResNet(BasicBlock, [3, 3, 3], cprate=cprate)


def resnet32(cprate):
    return ResNet(BasicBlock, [5, 5, 5], cprate=cprate)


def resnet44(cprate):
    return ResNet(BasicBlock, [7, 7, 7], cprate=cprate)


def resnet56(cprate):
    return ResNet(BasicBlock, [9, 9, 9], cprate=cprate)


def resnet110(cprate):
    return ResNet(BasicBlock, [18, 18, 18], cprate=cprate)


def resnet1202(cprate):
    return ResNet(BasicBlock, [200, 200, 200], cprate=cprate)


