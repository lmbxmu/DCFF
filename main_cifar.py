import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import thop.profile as profile

import data.cifar10 as cifar10
import utils.common as utils

from models.cifar import *

from utils.options import args

#*
import numpy as np
from collections import OrderedDict
from thop import profile
from scipy.spatial.distance import cdist



# Init
device = torch.device(f'cuda:{args.gpus[0]}') if torch.cuda.is_available() else 'cpu'
checkpoint = utils.CheckPoint(args)
logger = utils.GetLogger('DFF', os.path.join(args.job_dir + '/logger.log'))
logger_printP = utils.GetLogger('printP', os.path.join(args.job_dir + '/pringP.log'))
loss_func = nn.CrossEntropyLoss()


# Data
print('==> Preparing data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
else:
    raise NotImplementedError

default_cprate={
    #* vgg16-ABCPlus
    # 'vgg16':    [0.2]*3+[0.4]+[0.7]+[0.2]+[0.8]+[0.7]+[0.8]+[0.7]+[0.2]+[0.8]+[0.9],

    #* vgg16-HRankPlus
    # 'vgg16':    [0.21]*7+[0.75]*5+[0.0],
    # 'vgg16':    [0.3]*7+[0.75]*5+[0.0],
    'vgg16':    [0.45]*7+[0.78]*5+[0.0],

    #* vgg16-TEST
    # 'vgg16':    [0.5]*13,


    #* ========== resnet56 ====================
    #* len(cprate)=(9+9+9)+(3)=30
    #* (9+9+9) is stage1~3's conv1 cprate
    #* (3) is stage1~3's conv2's cprate, every blocks' conv2's cprate in the same stage should be the same.
    #* pre-conv1's cprate is the same as stage1's conv2's cprate
    #* ===== resnet56-ABCPlus =====
    'resnet56': [0.6]+[0.7]+[0.5]+[0.5]+[0.4]+[0.2]+[0.3]+[0.4]+[0.8]+
                [0.7]+[0.6]+[0.9]+[0.8]+[0.9]+[0.8]+[0.4]+[0.2]+[0.2]+
                [0.7]+[0.3]+[0.8]+[0.4]+[0.3]+[0.7]+[0.2]+[0.4]+[0.8]+
                [0.0]+[0.0]+[0.0],

    #* ===== resnet56-HRankPlus =====
    # 'resnet56':    [0.]+[0.18]*29,
    # 'resnet56':    [0.]+[0.15]*2+[0.4]*27,
    # 'resnet56':    [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9,

    #* ===== resnet56-TEST =====
    # 'resnet56': [0.1]*9 + [0.2]*9 + [0.3]*9 + [0.4]+[0.5]+[0.6],


    #* ========== resnet110 ====================
    #* len(cprate)=(18+18+18)+(3)=57
    #* (18+18+18) is stage1~3's conv1 cprate
    #* (3) is stage1~3's conv2's cprate, every blocks' conv2's cprate in the same stage should be the same.
    #* pre-conv1's cprate is the same as stage1's conv2's cprate
    #* ===== resnet110-ABCPlus =====
    'resnet110':[0.2]+[0.0]+[0.2]+[0.3]+[0.6]+[0.7]+[0.1]+[0.3]+[0.3]+[0.4]+[0.7]+[0.7]+[0.5]+[0.1]+[0.3]+[0.0]+[0.6]+[0.0]+
                [0.2]+[0.5]+[0.0]+[0.6]+[0.7]+[0.5]+[0.7]+[0.7]+[0.3]+[0.4]+[0.0]+[0.3]+[0.1]+[0.5]+[0.0]+[0.1]+[0.0]+[0.7]+
                [0.0]+[0.1]+[0.3]+[0.3]+[0.3]+[0.1]+[0.2]+[0.5]+[0.7]+[0.2]+[0.4]+[0.7]+[0.5]+[0.7]+[0.7]+[0.7]+[0.5]+[0.1]+
                [0.6]+[0.2]+[0.5],

    #* ===== resnet110-HRankPlus =====
    # 'resnet110':[0.]+[0.2]*2+[0.3]*18+[0.35]*36,
    # 'resnet110':[0.]+[0.25]*2+[0.4]*18+[0.55]*36,
    # 'resnet110':[0.]+[0.4]*2+[0.5]*18+[0.65]*36,

    #* ===== resnet110-TEST =====
    # 'resnet110':[0.1]*18 + [0.2]*18 + [0.3]*18 + [0.4]+[0.5]+[0.6],


    #* ========== googlenet ====================
    #* len(cprate)=(1) + (7+7+7+7+7+7+7+7+7)
    #* (1) is pre-conv1's cprate
    #* (7+7+7+7+7+7+7+7+7) is block1~9'cprate, every block has 7 conv layers.
    #* ===== googlenet-ABCPlus =====
    'googlenet':[0.0]+

                [0.0]+ [0.8]+[0.0] +[0.8]+[0.8]+[0.0] + [0.0]+

                [0.0]+ [0.9]+[0.0] +[0.9]+[0.9]+[0.0] + [0.0]+
                [0.0]+ [0.9]+[0.0] +[0.9]+[0.9]+[0.0] + [0.0]+
                [0.0]+ [0.9]+[0.0] +[0.9]+[0.9]+[0.0] + [0.0]+

                [0.0]+ [0.8]+[0.0] +[0.8]+[0.8]+[0.0] + [0.0]+
                [0.0]+ [0.8]+[0.0] +[0.8]+[0.8]+[0.0] + [0.0]+
                [0.0]+ [0.8]+[0.0] +[0.8]+[0.8]+[0.0] + [0.0]+

                [0.0]+ [0.9]+[0.0] +[0.9]+[0.9]+[0.0] + [0.0]+
                [0.0]+ [0.9]+[0.0] +[0.9]+[0.9]+[0.0] + [0.0],


    #* ===== googlenet-HRankPlus =====
    # 'googlenet':[0.3]+[0.6]*2+[0.7]*5+[0.8]*2,
    # 'googlenet':[0.4]+[0.85]*2+[0.9]*5+[0.9]*2,

    #* ===== googlenet-TEST =====
    # 'googlenet':[0.0]+[0.1]+[0.2]+[0.3]+[0.4]+[0.5]+[0.6]+[0.7]+[0.8]+[0.9],    
}



#* Compute cprate
if args.cprate:
    cprate = eval(args.cprate)
else:
    cprate = cprate = default_cprate[args.arch]

logger.info(f'cprate: \n{cprate}')


#* Compute resnet lawer-wise cprate

layer_wise_cprate = []
if 'vgg' in args.arch:
    layer_wise_cprate = cprate

elif 'resnet' in args.arch:
    resnet_block_num = {
        'resnet56': 9,
        'resnet110':18,
    }
    block_conv2_cprate = [val for val in cprate[-3:] for i in range(resnet_block_num[args.arch])]
    for item in zip(cprate[0:-3], block_conv2_cprate):
        layer_wise_cprate += list(item)
    layer_wise_cprate.insert(0, cprate[-3])

elif 'googlenet' == args.arch:
    # block_cprate = [val for val in cprate[1:] for i in range(7)]
    # layer_wise_cprate = cprate[0:1]+block_cprate
    layer_wise_cprate = cprate
else:
    raise NotImplementedError
print(f'layer-wise cprate: \n{layer_wise_cprate}')


# Model
print('==> Building model..')
if args.arch == 'vgg16':
    model = Fused_VGG(vgg_name='vgg16', cprate=layer_wise_cprate)
    compact_model = Compact_VGG(vgg_name='vgg16', cprate=layer_wise_cprate)
    origin_model = OriginVGG(vgg_name='vgg16')

elif args.arch == 'resnet56':
    model = fused_resnet56(cprate=layer_wise_cprate)
    compact_model = compact_resnet56(cprate=layer_wise_cprate)
    origin_model = origin_resnet56()

elif args.arch == 'resnet110':
    model = fused_resnet110(cprate=layer_wise_cprate)
    compact_model = compact_resnet110(cprate=layer_wise_cprate)
    origin_model = origin_resnet110()

elif args.arch == 'googlenet':
    model = FusedGoogLeNet(layer_wise_cprate)
    compact_model = CompactGoogLeNet(layer_wise_cprate)
    origin_model = OriginGoogLeNet()
else:
    raise NotImplementedError

# print(model)
# exit(0)

#* Compute flops, flops, puring rate
# inputs = torch.randn(1, 3, 32, 32)
# compact_flops, compact_params = profile(compact_model, inputs=(inputs, ))
# origin_flops, origin_params = profile(origin_model, inputs=(inputs, ))

# flops_prate = (origin_flops-compact_flops)/origin_flops
# params_prate = (origin_params-compact_params)/origin_params
# logger.info(f'{args.arch}\'s baseline model: FLOPs={origin_flops/10**6:.2f}M (0.0%), Params={origin_params/10**6:.2f}M (0.0%)')
# logger.info(f'{args.arch}\'s pruned   model: FLOPs={compact_flops/10**6:.2f}M ({flops_prate*100:.2f}%), Params={compact_params/10**6:.2f}M ({params_prate*100:.2f}%)')
# exit(0)


model = model.to(device)


if len(args.gpus) != 1:
    model = nn.DataParallel(model, device_ids=args.gpus)


# Training function
def train(model, optimizer, trainLoader, args, epoch):
    print(f'\nEpoch: {epoch+1}')
    model.train()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10

    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainLoader):
        inputs, targets = inputs.to(device), targets.to(device)


        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(outputs, targets)
        accuracy.update(prec1[0], inputs.size(0))

        if batch_idx % print_freq == 0 and batch_idx != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                f'Epoch[{epoch}] ({batch_idx * args.train_batch_size} / {len(trainLoader.dataset)}):\t'
                f'Loss: {float(losses.avg):.4f}\t'
                f'Acc: {float(accuracy.avg):.2f}%\t\t'
                f'Time: {cost_time:.2f}s'
            )
            start_time = current_time


# Testing function
def test(model, testLoader):
    global best_acc
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accuracy.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            f'Test Loss: {float(losses.avg):.4f}\t Acc: {float(accuracy.avg):.2f}%\t\t Time: {(current_time - start_time):.2f}'
        )
    return accuracy.avg


# main function
def main():
    global model, compact_model, origin_model

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_decay_step, gamma=0.1)

    # Resume from checkpoint (Train from pre-train model)
    if args.resume:
        # Load ckpt file.
        print('==> Resuming from checkpoint file..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'
        resume_ckpt = torch.load(args.resume)

        state_dict = resume_ckpt['state_dict']

        if len(args.gpus) == 1:  # load model when use single gpu
            model.load_state_dict(state_dict)
        else:                    # load model when use multi-gpus
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' not in k:
                    k = 'module.'+k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k]=v
            model.load_state_dict(new_state_dict)

        optimizer.load_state_dict(resume_ckpt['optimizer'])
        scheduler.load_state_dict(resume_ckpt['scheduler'])
        start_epoch = resume_ckpt['epoch']
        best_acc = resume_ckpt['best_acc']


    #* Train from scratch
    else:
        start_epoch = 0
        best_acc = 0.0


    #* Test only
    if args.test_only:
        test(model, loader.testLoader)


    #* Train
    else:
        #* setup fused_conv_modules
        fused_conv_modules = []
        for name, module in model.named_modules():
            if isinstance(module, FuseConv2d):
                fused_conv_modules.append(module)

        #* setup conv_module.layerid / layers_cout
        layers_cout = []
        for layerid, module in enumerate(fused_conv_modules):
            layers_cout.append(module.out_channels)

        #* Compute layers_m
        layers_cout = np.asarray(layers_cout)
        layers_cprate = np.asarray(layer_wise_cprate)
        layers_m = (layers_cout * (1-layers_cprate)).astype(int)


        for epoch in range(start_epoch, start_epoch + args.num_epochs):
            #* Compute t, t tends to 0 as epochs increases to num_epochs.
            # t = 1 - epoch / args.num_epochs
            t=eval(args.t_expression)
            # t = 0.1

            #* Compute layeri_param / layeri_negaEudist / layeri_softmaxP / layeri_KL / layeri_iScore
            start = time.time()
            for layerid, module in enumerate(fused_conv_modules):
                print(layerid)

                param = module.weight

                #* Compute layeri_param
                layeri_param = torch.reshape(param.detach(), (param.shape[0], -1))      #* layeri_param.shape=[cout, cin, k, k], layeri_param[j] means filterj's weight.

                #* Compute layeri_negaEudist
                # layeri_negaEudist = torch.mul(torch.cdist(layeri_param, layeri_param, p=2), -1)     #* layeri_negaEudist.shape=[cout, cout], layeri_negaEudist[j, k] means the negaEudist between filterj ans filterk.
                layeri_negaEudist = -torch.from_numpy(cdist(layeri_param.cpu(), layeri_param.cpu(), metric='euclidean').astype(np.float32)).to(device)

                #* Compute layeri_softmaxP
                softmax = nn.Softmax(dim=1)
                layeri_softmaxP = softmax(torch.div(layeri_negaEudist, t))      #* layeri_softmaxP.shape=[cout, cout], layeri_softmaxP[j] means filterj's softmax vector P.

                #* Compute layeri_KL
                layeri_KL = torch.sum(layeri_softmaxP[:,None,:] * (layeri_softmaxP[:,None,:]/(layeri_softmaxP+10**-7)).log(), dim = 2)      #* layeri_KL.shape=[cout, cout], layeri_KL[j, k] means KL divergence between filterj and filterk

                #* Compute layeri_iScore
                layeri_iScore = torch.sum(layeri_KL, dim=1)        #* layeri_iScore.shape=[cout], layeri_iScore[j] means filterj's importance score
                
                #* setup conv_module.layeri_topm_filters_id
                _, topm_ids = torch.topk(layeri_iScore, layers_m[layerid])

                _, topm_ids_order = torch.topk(layeri_iScore, layers_m[layerid], sorted=False)
                # print(topm_ids, topm_ids_order)
                # exit(0)

                logger_printP.info(f'Eopch: {epoch}, \t Layerid: {layerid}, \t topm_ids: {topm_ids.tolist()}, \ntopm_ids_iScore: {layeri_iScore[topm_ids].tolist()}')

                ##* setup conv_module.layeri_softmaxP
                module.layeri_softmaxP = layeri_softmaxP[topm_ids_order, :]

                ##* setup conv_module.layeri_onehotP
                # module.layeri_softmaxP = torch.eye(param.shape[0]).cuda()[topm_ids_order, :]


                ###* printP
                bottom_num = layers_cout[layerid]-layers_m[layerid]

                # if bottom_num > 0:
                #     _, bottom_ids = torch.topk(layeri_iScore, bottom_num, largest=False)
                
            
            print(f'cost: {time.time()-start:.2f}s')
            del param, layeri_param, layeri_negaEudist, layeri_KL, layeri_iScore, topm_ids

            train(model, optimizer, loader.trainLoader, args, epoch)
            scheduler.step()
            test_acc = float(test(model, loader.testLoader))

            is_best = best_acc < test_acc
            best_acc = max(best_acc, test_acc)

            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            
            state = {
                'state_dict': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_acc': best_acc,
            }
            checkpoint.save_model(state, epoch + 1, is_best)

            if is_best:
                #* Compute best compact_state_dict
                compact_state_dict = OrderedDict()
                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        if isinstance(module, FuseConv2d):
                            compact_state_dict[name+'.weight'] = module.fused_weight
                            if module.bias is not None:
                                compact_state_dict[name+'.bias'] = module.fused_bias
                        else:
                            compact_state_dict[name+'.weight'] = module.weight
                            if module.bias is not None:
                                compact_state_dict[name+'.bias'] = module.bias
                    if isinstance(module, nn.BatchNorm2d):
                        compact_state_dict[name+'.weight'] = module.weight
                        compact_state_dict[name+'.bias'] = module.bias
                        compact_state_dict[name+'.running_mean'] = module.running_mean
                        compact_state_dict[name+'.running_var'] = module.running_var
                        compact_state_dict[name+'.num_batches_tracked'] = module.num_batches_tracked
                    if isinstance(module, nn.Linear):
                        compact_state_dict[name+'.weight'] = module.weight
                        compact_state_dict[name+'.bias'] = module.bias
                
                #* Save best compact_state_dict
                checkpoint.save_compact_model(compact_state_dict)

        logger.info(f'Best model accuracy: {best_acc:.2f}')


        #* Test compact model
        compact_model = compact_model.to(device)
        
        compact_state_dict = torch.load(f'{args.job_dir}/checkpoint/model_best_compact.pt')
        compact_model.load_state_dict(compact_state_dict)
        
        compact_test_acc = float(test(compact_model, loader.testLoader))
        logger.info(f'Best Compact model accuracy:{compact_test_acc:.2f}%')

        #* Compute flops, flops, puring rate
        inputs = torch.randn(1, 3, 32, 32)
        compact_model = compact_model.cpu()
        compact_flops, compact_params = profile(compact_model, inputs=(inputs, ))
        origin_flops, origin_params = profile(origin_model, inputs=(inputs, ))

        flops_prate = (origin_flops-compact_flops)/origin_flops
        params_prate = (origin_params-compact_params)/origin_params
        logger.info(f'{args.arch}\'s baseline model: FLOPs = {origin_flops/10**6:.2f}M (0.0%), Params = {origin_params/10**6:.2f}M (0.0%)')
        logger.info(f'{args.arch}\'s pruned   model: FLOPs = {compact_flops/10**6:.2f}M ({flops_prate*100:.2f}%), Params = {compact_params/10**6:.2f}M ({params_prate*100:.2f}%)')



if __name__ == '__main__':
    main()
