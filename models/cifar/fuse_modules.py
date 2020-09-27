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
        self.layeri_topm_filters_id = []
        self.layeri_softmaxP = torch.zeros(1).cuda()

    def forward(self, input):
        cout = self.weight.shape[0]
        cin = self.weight.shape[1]
        k = self.weight.shape[2]

        # for filteri in range(cout):
        #     if filteri == 0:
        #         fused_filteri_weight = (self.weight.reshape(cout, -1) * self.layeri_softmaxP[filteri].unsqueeze(dim=1).cuda()).sum(0).unsqueeze(dim=0)
        #         fused_layeri_weight = fused_filteri_weight

        #         fused_filteri_bias = (self.bias * self.layeri_softmaxP[filteri].cuda()).sum(0).unsqueeze(dim=0)
        #         fused_layeri_bias = fused_filteri_bias

        #     else:
        #         fused_filteri_weight = (self.weight.reshape(cout, -1) * self.layeri_softmaxP[filteri].unsqueeze(dim=1).cuda()).sum(0).unsqueeze(dim=0)
        #         fused_layeri_weight = torch.cat((fused_layeri_weight, fused_filteri_weight), 0)

        #         fused_filteri_bias = (self.bias * self.layeri_softmaxP[filteri].cuda()).sum(0).unsqueeze(dim=0)
        #         fused_layeri_bias = torch.cat((fused_layeri_bias, fused_filteri_bias), 0)
        # print(self.layeri_softmaxP)
        # print(self.layeri_softmaxP.unsqueeze(1))
        # print(self.layeri_softmaxP.unsqueeze(1).expand(-1, 27, -1).shape)
        # print(self.weight.reshape(cout, -1).unsqueeze(2).expand(-1, -1, cout).shape)
        # exit(0)
        fused_layeri_weight = (self.weight.reshape(cout, -1).unsqueeze(2).expand(-1, -1, cout) * self.layeri_softmaxP.unsqueeze(1).expand(-1, cin*k*k, -1)).sum(dim=2)

        fused_layeri_weight = torch.index_select(fused_layeri_weight, 0, self.layeri_topm_filters_id)

        # print(fused_layeri_weight.shape)
        fused_layeri_weight = fused_layeri_weight.reshape(cout, cin, k, k)

        # set the weight (not in top m) to zero
        # for filters in range(self.weight.shape[0]):
        #     if filters not in self.layeri_topm_filters_id:
        #         # with torch.no_grad():
        #         fused_layeri_weight[filters].mul_(0)
        #         fused_layeri_bias[filters].mul_(0)



        output = F.conv2d(input=input, weight=fused_layeri_weight, bias=fused_layeri_bias, stride= self.stride, padding= self.padding, dilation= self.dilation, groups= self.groups)
        exit(0)
        return output
