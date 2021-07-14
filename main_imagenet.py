import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

import data.imagenet as imagenet
import utils.common as utils

from models.imagenet import *

from utils.options import args

import numpy as np
from collections import OrderedDict
from thop import profile
from scipy.spatial.distance import cdist
import math


#* Init
visible_gpus_str = ','.join(str(i) for i in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus_str
args.gpus = [i for i in range(len(args.gpus))]

device = torch.device(f'cuda:{args.gpus[0]}') if torch.cuda.is_available() else 'cpu'
checkpoint = utils.CheckPoint(args)
logger = utils.GetLogger('DFF', os.path.join(args.job_dir + '/logger.log'))
loss_func = nn.CrossEntropyLoss()


#* Data
print('==> Preparing data..')
if args.data_set == 'imagenet':
        data_tmp = imagenet.Data(args)
        trainLoader = data_tmp.trainLoader
        testLoader = data_tmp.testLoader
else:
    raise NotImplementedError


default_cprate={
    'resnet50': [0.0]*37,
}

#* Compute cprate
if args.cprate:
    cprate = eval(args.cprate)
else:
    cprate = default_cprate[args.arch]

logger.info(f'{args.arch}\'s cprate: \n{cprate}')


#* Compute resnet lawer-wise cprate
layer_wise_cprate = []
if 'resnet' in args.arch:
    resnet_block_num = {
        'resnet50': [3,4,6,3],
    }
    total_block_num = sum(resnet_block_num[args.arch])

    pre_conv1_cprate = cprate[0]

    block_conv1_cprates = cprate[1 : 1+total_block_num]
    print(block_conv1_cprates)

    block_conv2_cprates = cprate[1+total_block_num : 1+total_block_num*2]
    print(block_conv2_cprates)

    block_conv3_cprates = []
    for stageid, block_conv3_cprate in enumerate(cprate[-4:]):
        for i in range(resnet_block_num[args.arch][stageid]):
            block_conv3_cprates.append(block_conv3_cprate)
    print(block_conv3_cprates)

    for item in zip(block_conv1_cprates, block_conv2_cprates, block_conv3_cprates):
        layer_wise_cprate += list(item)
    layer_wise_cprate.insert(0, cprate[0])
else:
    raise NotImplementedError
print(f'layer-wise cprate: \n{layer_wise_cprate}')


#* Model
print('==> Building model..')
if args.arch == 'resnet50':
    model = fused_resnet50(cprate=layer_wise_cprate)
    compact_model = compact_resnet50(cprate=layer_wise_cprate)
    origin_model = origin_resnet50()
else:
    raise NotImplementedError

#* Compute flops, flops, puring rate
if args.get_flops:
    inputs = torch.randn(1, 3, 224, 224)
    compact_model = compact_model.cpu()
    origin_model = origin_model.cpu()
    compact_flops, compact_params = profile(compact_model, inputs=(inputs, ))
    origin_flops, origin_params = profile(origin_model, inputs=(inputs, ))

    flops_prate = (origin_flops-compact_flops)/origin_flops
    params_prate = (origin_params-compact_params)/origin_params
    logger.info(f'{args.arch}\'s baseline model: FLOPs={origin_flops/10**6:.2f}M (0.0%), Params={origin_params/10**6:.2f}M (0.0%)')
    logger.info(f'{args.arch}\'s pruned   model: FLOPs={compact_flops/10**6:.2f}M ({flops_prate*100:.2f}%), Params={compact_params/10**6:.2f}M ({params_prate*100:.2f}%)')
    exit(0)

model = model.to(device)

if len(args.gpus) != 1:
    model = nn.DataParallel(model, device_ids=args.gpus)


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    if args.lr_type == 'step':
        factor = epoch // 30
        if epoch >= 80:
            factor = factor + 1
        lr = args.lr * (0.1 ** factor)
    elif args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.num_epochs - 5)))
    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError

    # Warmup
    if epoch < 5:
            lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)
    if step == 0:
        print('current learning rate:{0}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training function
def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):
    print(f'\nEpoch: {epoch}')
    model.train()

    losses = utils.AverageMeter()
    top1_accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    
    start_time = time.time()


    num_iter = len(trainLoader)
    print_freq = num_iter // 10

    for batch_idx, (inputs, targets) in enumerate(trainLoader):
        if args.debug:
            if batch_idx > 5:
                break
        inputs, targets = inputs.to(device), targets.to(device)
        adjust_learning_rate(optimizer, epoch, batch_idx, num_iter)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        pred = utils.accuracy(outputs, targets, topk=topk)
        top1_accuracy.update(pred[0], inputs.size(0))
        top5_accuracy.update(pred[1], inputs.size(0))

        if (batch_idx % print_freq == 0 and batch_idx != 0) or args.debug:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                f'Epoch[{epoch}] ({batch_idx} / {num_iter}):\t'
                f'Loss: {float(losses.avg):.6f}\t'
                f'Top1: {float(top1_accuracy.avg):.6f}%\t'
                f'Top5: {float(top5_accuracy.avg):.6f}%\t'
                f'Time: {cost_time:.2f}s'
            )
            start_time = current_time


#* Testing function
def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    top1_accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            if args.debug:
                if batch_idx > 5:
                    break
            inputs, targets = inputs.to(device), targets.to(device)

            #* compute output
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            #* measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))
            pred = utils.accuracy(outputs, targets, topk=topk)
            top1_accuracy.update(pred[0], inputs.size(0))
            top5_accuracy.update(pred[1], inputs.size(0))

        #* measure elapsed time
        current_time = time.time()
        logger.info(
            f'Test Loss: {float(losses.avg):.6f}\t Top1: {float(top1_accuracy.avg):.6f}%\t'
            f'Top5: {float(top5_accuracy.avg):.6f}%\t Time: {float(current_time - start_time):.2f}s'
        )

    return float(top1_accuracy.avg), float(top5_accuracy.avg)


#* main function
def main():
    global model, compact_model, origin_model

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    #* Resume from checkpoint (Train from pre-train model)
    if args.resume:
        #* Load ckpt file.
        args.resume = args.job_dir+'/checkpoint/model_last.pt'
        print(f'==> Loading cheakpoint {args.resume}......')
        assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'
        
        resume_ckpt = torch.load(args.resume, map_location=device)

        optimizer.load_state_dict(resume_ckpt['optimizer'])
        start_epoch = resume_ckpt['epoch'] + 1
        best_top1_acc = resume_ckpt['best_top1_acc']
        best_top5_acc = resume_ckpt['best_top5_acc']
        layers_iScore_resume = resume_ckpt['layers_iScore']

        tmp_state_dict = resume_ckpt['state_dict']
        new_state_dict = OrderedDict()
        if len(args.gpus) > 1:
            for k, v in tmp_state_dict.items():
                new_state_dict['module.' + k.replace('module.', '')] = v
                model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(tmp_state_dict)

    
    #* Train from scratch
    else:
        start_epoch = 0
        best_top1_acc = 0.0
        best_top5_acc = 0.0


    #* Test only
    if args.test_only:
        logger.info(model)
        logger.info(compact_model)
        logger.info(model.state_dict().keys())
        test(model, testLoader, topk=(1, 5))
        
    #* Train
    else:
        #* setup fused_conv_modules
        model_name_modules = model.module.named_modules() if len(args.gpus) > 1 else model.named_modules()
        fused_conv_modules = []
        for name, module in model_name_modules:
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

        layers_iScore = []
        for epoch in range(start_epoch, args.num_epochs):

            print(f"\nComputing epoch {epoch}'s all fused layer...")
            start = time.time()
            for layerid, module in enumerate(fused_conv_modules):

                param = module.weight

                #* Compute layeri_param
                layeri_param = torch.reshape(param.detach(), (param.shape[0], -1))      #* layeri_param.shape=[cout, cin, k, k], layeri_param[j] means filterj's weight.

                #* Compute layeri_Eudist
                layeri_Eudist = cdist(layeri_param.cpu(), layeri_param.cpu(), metric='euclidean').astype(np.float32)


                 #* Compute layeri_negaEudist
                layeri_negaEudist = -torch.from_numpy(layeri_Eudist).to(device)

                #* Compute layeri_softmaxP
                softmax = nn.Softmax(dim=1)

                #* Compute t
                Ts = 1
                Te = 10000
                e = epoch
                E = args.num_epochs
                pi = math.pi

                k = 1
                A = 2*(Te-Ts)*(1+math.exp(-k*E)) / (1-math.exp(-k*E))
                T = A/(1+math.exp(-k*e)) + Ts - A/2
                t = 1/T

                layeri_softmaxP = softmax(layeri_negaEudist / t)        #* layeri_softmaxP.shape=[cout, cout], layeri_softmaxP[j] means filterj's softmax vector P.

                #* Compute layeri_KL
                try:
                    layeri_KL = torch.mean(layeri_softmaxP[:,None,:] * (layeri_softmaxP[:,None,:]/(layeri_softmaxP+1e-7)).log(), dim = 2)      #* layeri_KL.shape=[cout, cout], layeri_KL[j, k] means KL divergence between filterj and filterk
                except:
                    layeri_softmaxP = layeri_softmaxP.cpu()
                    layeri_KL = torch.sum(layeri_softmaxP[:,None,:] * (layeri_softmaxP[:,None,:]/layeri_softmaxP).log(), dim = 2)      #* layeri_KL.shape=[cout, cout], layeri_KL[j, k] means KL divergence between filterj and filterk
                    layeri_softmaxP = layeri_softmaxP.to(device)
                    layeri_KL = layeri_KL.to(device)

                #* Compute layers_iScore
                layeri_iScore_kl = torch.sum(layeri_KL, dim=1)
                if epoch == start_epoch:
                    if epoch == 0:
                        layers_iScore.append(layeri_iScore_kl)
                    else:
                        layers_iScore = layers_iScore_resume
                else:
                    if False not in torch.isfinite(layeri_iScore_kl):
                        layers_iScore[layerid] = layeri_iScore_kl        #* layers_iScore.shape=[cout], layers_iScore[j] means filterj's importance score
                    else:
                        pass

                #* setup conv_module.layeri_topm_filters_id
                _, topm_ids = torch.topk(layers_iScore[layerid], int(layers_m[layerid]))
                _, topm_ids_order = torch.topk(layers_iScore[layerid], int(layers_m[layerid]), sorted=False)

                #* Compute layeri_p
                softmaxP = layeri_softmaxP[topm_ids_order, :]
                onehotP = torch.eye(param.shape[0]).to(device)[topm_ids_order, :]



                #* setup conv_module.layeri_softmaxP
                module.layeri_softmaxP = softmaxP
                # module.layeri_softmaxP = onehotP
                del param, layeri_param, layeri_negaEudist, layeri_KL

            print(f'\nEpoch {epoch} fuse all layers cost: {time.time()-start:.2f}s')

            train(model, optimizer, trainLoader, args, epoch, topk=(1, 5))
            test_top1_acc, test_top5_acc = test(model, testLoader, topk=(1, 5))
            
            is_best = best_top1_acc < test_top1_acc
            if is_best:
                best_top1_acc = max(best_top1_acc, test_top1_acc)
                best_top5_acc = max(best_top5_acc, test_top5_acc)
                logger.info(f'best acc top1: {float(best_top1_acc):.6f}%, top5: {float(best_top5_acc):.6f}%')

            
            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            
            state = {
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_top1_acc': best_top1_acc,
                'best_top5_acc': best_top5_acc,
                'state_dict': model_state_dict,
                'layers_iScore' : layers_iScore,
            }
            checkpoint.save_model(state, epoch, is_best)

            if is_best or args.debug:
                #* Compute best compact_state_dict
                compact_state_dict = OrderedDict()
                model_name_modules = model.module.named_modules() if len(args.gpus) > 1 else model.named_modules()
                for name, module in model_name_modules:
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

        logger.info(f'Best Acc Top1: {float(best_top1_acc):.6f}%, \tTop5: {float(best_top5_acc):.6f}%')

        #* Test compact model
        compact_model = compact_model.to(device)
        compact_state_dict = torch.load(f'{args.job_dir}/checkpoint/model_best_compact.pt')
        compact_model.load_state_dict(compact_state_dict)
        
        compact_test_acc = test(compact_model, testLoader, topk=(1, 5))
        logger.info(f'Best Compact Model Acc Top1: {compact_test_acc[0]:.6f}%, Top5: {compact_test_acc[1]:.6f}%')

        #* Compute flops, flops, puring rate
        inputs = torch.randn(1, 3, 224, 224)
        compact_model = compact_model.cpu()
        origin_model = origin_model.cpu()
        compact_flops, compact_params = profile(compact_model, inputs=(inputs, ))
        origin_flops, origin_params = profile(origin_model, inputs=(inputs, ))

        flops_prate = (origin_flops-compact_flops)/origin_flops
        params_prate = (origin_params-compact_params)/origin_params
        logger.info(f'{args.arch}\'s baseline model: FLOPs = {origin_flops/10**6:.2f}M (0.0%), Params = {origin_params/10**6:.2f}M (0.0%)')
        logger.info(f'{args.arch}\'s pruned   model: FLOPs = {compact_flops/10**9:.2f}B ({flops_prate*100:.2f}%), Params = {compact_params/10**6:.2f}M ({params_prate*100:.2f}%)')


if __name__ == '__main__':
    main()
