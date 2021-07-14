import torch
import torch.nn as nn

from .fuse_modules import FuseConv2d


#*================================
#* fused model
def fused_conv3x3(in_planes, out_planes, real_cout, stride=1):
    """3x3 convolution with padding"""
    return FuseConv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1, bias=False, real_cout=real_cout)

def fused_conv1x1(in_planes, out_planes, real_cout, stride=1):
    """1x1 convolution"""
    return FuseConv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False, real_cout=real_cout)


class FusedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, cprate, cur_convid, stride=1, downsample=None):
        super(FusedBasicBlock, self).__init__()

        last_cout = int(in_planes*(1-cprate[cur_convid-1]))
        cur_cout = int(planes*(1-cprate[cur_convid]))

        self.conv1 = fused_conv1x1(in_planes=last_cout, out_planes=planes, stride=stride, real_cout=cur_cout)
        self.bn1 = nn.BatchNorm2d(cur_cout)
        self.relu = nn.ReLU(inplace=True)

        cur_convid = cur_convid + 1

        last_cout = cur_cout
        cur_cout = int(planes*(1-cprate[cur_convid]))

        self.conv2 = fused_conv3x3(in_planes=last_cout, out_planes=planes, real_cout=cur_cout)
        self.bn2 = nn.BatchNorm2d(cur_cout)

        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FusedBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, cprate, cur_convid, stride=1, downsample=None):
        super(FusedBottleneck, self).__init__()

        last_cout = int(in_planes*(1-cprate[cur_convid-1]))
        cur_cout = int(planes*(1-cprate[cur_convid]))
        self.conv1 = fused_conv1x1(in_planes=last_cout, out_planes=planes, real_cout=cur_cout)
        self.bn1 = nn.BatchNorm2d(cur_cout)
        self.relu = nn.ReLU(inplace=True)

        cur_convid = cur_convid + 1

        last_cout = cur_cout
        cur_cout = int(planes*(1-cprate[cur_convid]))
        self.conv2 = fused_conv3x3(in_planes=last_cout, out_planes=planes, stride=stride, real_cout=cur_cout)
        self.bn2 = nn.BatchNorm2d(cur_cout)

        cur_convid = cur_convid + 1

        last_cout = cur_cout
        cur_cout = int(self.expansion * planes * (1-cprate[cur_convid]))

        self.conv3 = fused_conv1x1(in_planes=last_cout, out_planes=self.expansion * planes, real_cout=cur_cout)
        self.bn3 = nn.BatchNorm2d(cur_cout)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class FusedResNet(nn.Module):
    def __init__(self, block, layers, cprate=None, num_classes=1000):
        super(FusedResNet, self).__init__()
        #* 
        self.cprate = cprate
        self.cur_convid = 0

        if block==FusedBasicBlock:
            self.block_conv_num = 2
        elif block==FusedBottleneck:
            self.block_conv_num = 3

        self.in_planes = 64

        #*
        cur_cout = int(64*(1-cprate[self.cur_convid]))
        self.conv1 = FuseConv2d(in_channels=3,out_channels=64, kernel_size=7, stride=2, padding=3, bias=False, real_cout=cur_cout)
        self.bn1 = nn.BatchNorm2d(cur_cout)
        self.relu = nn.ReLU(inplace=True)

        #*
        self.cur_convid += 1

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        cur_cout = int(512 * block.expansion * (1-self.cprate[self.cur_convid-1]))
        self.fc = nn.Linear(cur_cout, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            last_cout = int(self.in_planes * (1-self.cprate[self.cur_convid-1]))
            cur_cout = int(planes * block.expansion * (1-self.cprate[self.cur_convid+2]))

            downsample = nn.Sequential(
                #*
                nn.Conv2d(last_cout, cur_cout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(cur_cout),
            )

        layers = []
        #*
        layers.append(block(self.in_planes, planes, cprate=self.cprate, cur_convid=self.cur_convid, stride=stride, downsample=downsample))
        self.cur_convid += self.block_conv_num

        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, cprate=self.cprate, cur_convid=self.cur_convid))
            self.cur_convid += self.block_conv_num

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def fused_resnet18(cprate):
    return FusedResNet(FusedBasicBlock, [2,2,2,2], cprate)

def fused_resnet34(cprate):
    return FusedResNet(FusedBasicBlock, [3,4,6,3], cprate)

def fused_resnet50(cprate):
    return FusedResNet(FusedBottleneck, [3,4,6,3], cprate)

def fused_resnet101(cprate):
    return FusedResNet(FusedBottleneck, [3,4,23,3], cprate)

def fused_resnet152(cprate):
    return FusedResNet(FusedBottleneck, [3,8,36,3], cprate)









#*================================
#* compact model
def compact_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def compact_conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CompactBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, cprate, cur_convid, stride=1, downsample=None):
        super(CompactBasicBlock, self).__init__()

        last_cout = int(in_planes*(1-cprate[cur_convid-1]))
        cur_cout = int(planes*(1-cprate[cur_convid]))

        self.conv1 = compact_conv1x1(last_cout, cur_cout, stride)
        self.bn1 = nn.BatchNorm2d(cur_cout)
        self.relu = nn.ReLU(inplace=True)

        cur_convid = cur_convid + 1

        last_cout = cur_cout
        cur_cout = int(planes*(1-cprate[cur_convid]))

        self.conv2 = compact_conv3x3(last_cout, cur_cout)
        self.bn2 = nn.BatchNorm2d(cur_cout)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CompactBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, cprate, cur_convid, stride=1, downsample=None):
        super(CompactBottleneck, self).__init__()

        last_cout = int(in_planes*(1-cprate[cur_convid-1]))
        cur_cout = int(planes*(1-cprate[cur_convid]))
        self.conv1 = compact_conv1x1(last_cout, cur_cout)
        self.bn1 = nn.BatchNorm2d(cur_cout)
        self.relu = nn.ReLU(inplace=True)

        cur_convid = cur_convid + 1

        last_cout = cur_cout
        cur_cout = int(planes*(1-cprate[cur_convid]))
        self.conv2 = compact_conv3x3(last_cout, cur_cout, stride)
        self.bn2 = nn.BatchNorm2d(cur_cout)

        cur_convid = cur_convid + 1

        last_cout = cur_cout
        cur_cout = int(self.expansion * planes * (1-cprate[cur_convid]))
        self.conv3 = compact_conv1x1(last_cout, cur_cout)
        self.bn3 = nn.BatchNorm2d(cur_cout)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class CompactResNet(nn.Module):
    def __init__(self, block, layers, cprate=None, num_classes=1000):
        super(CompactResNet, self).__init__()
        #* 
        self.cprate = cprate
        self.cur_convid = 0

        if block==CompactBasicBlock:
            self.block_conv_num = 2
        elif block==CompactBottleneck:
            self.block_conv_num = 3

        self.in_planes = 64

        #*
        cur_cout = int(64*(1-cprate[self.cur_convid]))
        self.conv1 = nn.Conv2d(3, cur_cout, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(cur_cout)
        self.relu = nn.ReLU(inplace=True)

        #*
        self.cur_convid += 1

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        cur_cout = int(512 * block.expansion * (1-self.cprate[self.cur_convid-1]))
        self.fc = nn.Linear(cur_cout, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:

            last_cout = int(self.in_planes * (1-self.cprate[self.cur_convid-1]))
            cur_cout = int(planes * block.expansion * (1-self.cprate[self.cur_convid+2]))
            downsample = nn.Sequential(
                #*
                nn.Conv2d(last_cout, cur_cout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(cur_cout),
            )

        layers = []
        #*
        layers.append(block(self.in_planes, planes, cprate=self.cprate, cur_convid=self.cur_convid, stride=stride, downsample=downsample))
        self.cur_convid += self.block_conv_num

        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, cprate=self.cprate, cur_convid=self.cur_convid))
            self.cur_convid += self.block_conv_num

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def compact_resnet18(cprate):
    return CompactResNet(CompactBasicBlock, [2,2,2,2], cprate)

def compact_resnet34(cprate):
    return CompactResNet(CompactBasicBlock, [3,4,6,3], cprate)

def compact_resnet50(cprate):
    return CompactResNet(CompactBottleneck, [3,4,6,3], cprate)

def compact_resnet101(cprate):
    return CompactResNet(CompactBottleneck, [3,4,23,3], cprate)

def compact_resnet152(cprate):
    return CompactResNet(CompactBottleneck, [3,8,36,3], cprate)



#*================================
#* origin model
def origin_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def origin_conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class OriginBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(OriginBasicBlock, self).__init__()

        self.conv1 = origin_conv1x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = origin_conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class OriginBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(OriginBottleneck, self).__init__()
        self.conv1 = origin_conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = origin_conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = origin_conv1x1(planes, self.expansion * planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class OriginResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(OriginResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def origin_resnet18():
    return OriginResNet(OriginBasicBlock, [2,2,2,2])

def origin_resnet34():
    return OriginResNet(OriginBasicBlock, [3,4,6,3])

def origin_resnet50():
    return OriginResNet(OriginBottleneck, [3,4,6,3])

def origin_resnet101():
    return OriginResNet(OriginBottleneck, [3,4,23,3])

def origin_resnet152():
    return OriginResNet(OriginBottleneck, [3,8,36,3])













