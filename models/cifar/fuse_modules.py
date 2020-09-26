import torch
import torch.nn as nn
import torch.nn.functional as F

class FuseConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(FuseConv2d, self).__init__(*kargs, **kwargs)
        #* layerid不是训练感知的, 只要在训练前初始化一次
        self.layerid = -1
        #* layer的训练感知参数(layer无关, 所有layer相同): epoch
        self.epoch = -1
        #* layer的训练感知参数(layer有关, 不同layer不同): layeri_topm_filters_id, layeri_softmaxP
        self.layeri_topm_filters_id = []
        self.layeri_softmaxP = torch.zeros(1).cuda()

    def forward(self, input):
        # print(f'self.layerid:{self.layerid}')
        # print(self.epoch)
        # print(f'self.layeri_topm_filters_id:{self.layeri_topm_filters_id}')
        # print(self.layeri_softmaxP, self.layeri_softmaxP.shape)
        # exit(0)

        #* 计算该层融合层的filters的新权重
        fused_filters_weight = self.weight

        fused_filters_bias = self.bias

        # 将融合层中不是topm的filters置零
        for layeri in range(fused_filters_weight.shape[0]):
            if layeri not in self.layeri_topm_filters_id:
                with torch.no_grad():
                    fused_filters_weight[layeri].mul_(0)
                    fused_filters_bias[layeri].mul_(0)

        output = F.conv2d(input=input, weight=fused_filters_weight, bias=fused_filters_bias, stride= self.stride, padding= self.padding, dilation= self.dilation, groups= self.groups)

        return output