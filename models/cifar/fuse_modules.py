import torch
import torch.nn as nn
import torch.nn.functional as F

class FuseConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(FuseConv2d, self).__init__(*kargs, **kwargs)
        #* setup non-training-aware attr.
        self.layerid = -1

        #* setup training-aware attr.
        ##* non-layer-aware
        self.epoch = -1
        ##* layer-aware
        self.layeri_softmaxP = torch.zeros(1).cuda()

    def forward(self, input):
        cout, cin, k, _ = self.weight.shape

        fused_layeri_weight = torch.mm(self.layeri_softmaxP, self.weight.reshape(cout, -1))
        fused_layeri_bias = torch.mm(self.layeri_softmaxP, self.bias.unsqueeze(1)).squeeze()

        fused_layeri_weight = fused_layeri_weight.reshape(-1, cin, k, k)

        output = F.conv2d(input=input, weight=fused_layeri_weight, bias=fused_layeri_bias, stride= self.stride, padding= self.padding, dilation= self.dilation, groups= self.groups)

        return output
