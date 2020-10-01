import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import thop.profile as profile

import data.imagenet as imagenet
import utils.common as utils

from models.imagenet import *

from utils.options import args

#*
import numpy as np
from collections import OrderedDict
from thop import profile



# init
device = torch.device(f'cuda:{args.gpus[0]}') if torch.cuda.is_available() else 'cpu'
checkpoint = utils.CheckPoint(args)
logger = utils.GetLogger(os.path.join(args.job_dir + '/logger.log'))
loss_func = nn.CrossEntropyLoss()


# Data
print('==> Preparing data..')
# global trainLoader, testLoader

if args.data_set == 'imagenet':
    # trainLoader = get_data_set('train')
    # testLoader = get_data_set('test')
    data_tmp = imagenet.Data(args)
    trainLoader = data_tmp.trainLoader
    testLoader = data_tmp.testLoader


default_cprate={
    ##*resnet50: len(cprate)=(1+(3+4+6+3)+4)=21, pre-conv1's cprate is the same as stage 1's conv2's cprate
    #* resnet50-ABCPlus
    # 'resnet50': [0.5]*9+[0.6]*11+[0.0],
    # 'resnet50': [0.4]*16+[0.5]*4+[0.0],
    # 'resnet50': [0.1]*5+[0.2]*15+[0.0],
    # 'resnet50': [0.4]*10+[0.5]*10+[0.0],

    #* resnet50-HRankPlus
    # 'resnet50':    [0.]+[0.35]*16+[0.1]*3+[0.0],
    # 'resnet50':    [0.]+[0.5]*16+[0.25]*3+[0.0],
    # 'resnet50':    [0.]+[0.6]*16+[0.5]*3+[0.0],

    #* resnet50-TEST
    'resnet50': [0.9] + [0.1]*3+[0.2]*4+[0.3]*6+[0.4]*3 + [0.5]+[0.6]+[0.7]+[0.8],



    #* mobilenetv1
    'mobilenetv1':[]
 
}

#* compute cprate
if args.cprate:
    cprate = eval(args.cprate)
else:
    cprate = cprate = default_cprate[args.arch]

logger.info(f'{args.arch}\'s cprate: \n{cprate}')

#* compute resnet lawer-wise cprate
resnet_block_num = {
    'resnet50': [3,4,6,3],
}
layer_wise_cprate = []
if 'resnet' in args.arch:
    block_conv1_cprates = list(map(float, [0] * sum(resnet_block_num[args.arch])))
    print(block_conv1_cprates)

    block_conv2_cprates = cprate[1:-4]
    print(block_conv2_cprates)

    block_conv3_cprates = []
    for stageid, block_conv3_cprate in enumerate(cprate[-4:]):
        for i in range(resnet_block_num[args.arch][stageid]):
            block_conv3_cprates.append(block_conv3_cprate)
    print(block_conv3_cprates)
    # exit(0)

    for item in zip(block_conv1_cprates, block_conv2_cprates, block_conv3_cprates):
        layer_wise_cprate += list(item)
    layer_wise_cprate.insert(0, cprate[0])

print(f'layer-wise cprate: \n{layer_wise_cprate}')

# Model
print('==> Building model..')
if args.arch == 'resnet50':
    model = fused_resnet50(cprate=layer_wise_cprate)

elif args.arch == 'mobilenetv1':
    model = mobilenetv1()

# print(model)
# exit(0)

model = model.to(device)

if len(args.gpus) != 1:
    model = nn.DataParallel(model, device_ids=args.gpus)


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    factor = epoch // 30
    if epoch >= 80:
        factor = factor + 1
    lr = args.lr * (0.1 ** factor)
    # Warmup
    # if epoch < 5:
    #     lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training function
def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):
    print(f'\nEpoch: {epoch+1}')
    model.train()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    print_freq = len(trainLoader) // args.train_batch_size // 10
    start_time = time.time()

    total = 0
    correct = 0


    for batch_idx, (inputs, targets) in enumerate(trainLoader):
        print(f'batch_idx:{batch_idx}')
        inputs, targets = inputs.to(device), targets.to(device)

        adjust_learning_rate(optimizer, epoch, batch_idx, len(trainLoader) // args.train_batch_size)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(outputs, targets, topk=topk)
        accuracy.update(prec1[0], inputs.size(0))
        top5_accuracy.update(prec1[1], inputs.size(0))

        if batch_idx % print_freq == 0 and batch_idx != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                f'Epoch[{epoch}] ({batch_idx * args.train_batch_size} / {len(trainLoader)}):\t'
                f'Loss: {float(losses.avg):.4f}\t'
                f'Top1: {float(accuracy.avg):.2f}%\t'
                f'Top5: {float(top5_accuracy.avg):.2f}%\t'
                f'Time: {cost_time:.2f}s'
            )
            start_time = current_time




# Testing function
def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testLoader):
            inputs = batch_data[0]['data'].to(device)
            targets = batch_data[0]['label'].squeeze().long().to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accuracy.update(predicted[0], inputs.size(0))
            top5_accuracy.update(predicted[1], inputs.size(0))

            # Debug
            current_time = time.time()
            logger.info(
                f'Test Loss: {float(losses.avg):.4f}\t Top1: {float(accuracy.avg):.2f}%\t'
                f'Top5: {float(accuracy.avg):.2f}%\t Time: {float(current_time - start_time):.2f}s'
            )

        current_time = time.time()
        logger.info(
            f'Test Loss: {float(losses.avg):.4f}\t Top1: {float(accuracy.avg):.2f}%\t'
            f'Top5: {float(accuracy.avg):.2f}%\t Time: {float(current_time - start_time):.2f}s'
        )
        
    testLoader.reset()
    return float(accuracy.avg), float(top5_accuracy.avg)

# main function
def main():
    global model

    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume from checkpoint (Train from pre-train model)
    if args.resume:
        # Load ckpt file.
        print('==> Resuming from checkpoint file..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'
        resume_ckpt = torch.load(args.resume, map_location=device)

        state_dict = resume_ckpt['state_dict']

        if len(args.gpus) == 1:  # load model when use single gpu
            model.load_state_dict(state_dict)
        else:                    # load model when use multi-gpus
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' not in k:
                    k = 'module.'+k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k]=v
            model.load_state_dict(new_state_dict)

        optimizer.load_state_dict(resume_ckpt['optimizer'])
        start_epoch = resume_ckpt['epoch']
        best_top1_acc = resume_ckpt['best_top1_acc']
        best_top5_acc = resume_ckpt['best_top5_acc']
    # Train from scratch
    else:
        start_epoch = 0
        best_top1_acc = 0.0
        best_top5_acc = 0.0


    # test only
    if args.test_only:
        test(model, testLoader, topk=(1, 5))
        
    # train
    else:
        #*
        fused_conv_modules = []
        for name, module in model.named_modules():
            if isinstance(module, FuseConv2d):
                fused_conv_modules.append(module)

        #* setup conv_module.layerid / layers_cout
        layers_cout = []
        for layerid, module in enumerate(fused_conv_modules):
            module.layerid = layerid
            layers_cout.append(module.out_channels)

        #* compute layers_m
        layers_cout = np.asarray(layers_cout)
        layers_cprate = np.asarray(layer_wise_cprate)
        layers_m = (layers_cout * (1-layers_cprate)).astype(int)
        print(layers_cout)
        print(layers_m)
        # exit(0)


        for epoch in range(start_epoch, start_epoch + args.num_epochs):
            #* compute t, t tends to 0 as epochs increases to num_epochs.
            # t = 1 - epoch / args.num_epochs
            t=eval(args.t_expression)
            #* compute layeri_param / layeri_negaEudist / layeri_softmaxP / layeri_KL / layeri_iScore
            start = time.time()
            for layerid, module in enumerate(fused_conv_modules):
                print(layerid)

                param = module.weight

                #* compute layeri_param
                layeri_param = torch.reshape(param.detach(), (param.shape[0], -1))      #* layeri_param.shape=[cout, cin, k, k], layeri_param[j] means filterj's weight.

                #* compute layeri_negaEudist
                layeri_negaEudist = torch.mul(torch.cdist(layeri_param, layeri_param, p=2), -1)     #* layeri_negaEudist.shape=[cout, cout], layeri_negaEudist[j, k] means the negaEudist between filterj ans filterk.

                #* compute layeri_softmaxP
                softmax = nn.Softmax(dim=1)
                layeri_softmaxP = softmax(torch.div(layeri_negaEudist, t))      #* layeri_softmaxP.shape=[cout, cout], layeri_softmaxP[j] means filterj's softmax vector P.

                #* compute layeri_KL
                print(layeri_softmaxP.shape)
                # layeri_softmaxP = layeri_softmaxP.cpu()
                
                layeri_KL = torch.mean(layeri_softmaxP[:,None,:] * (layeri_softmaxP[:,None,:]/layeri_softmaxP).log(), dim = 2)      #* layeri_KL.shape=[cout, cout], layeri_KL[j, k] means KL divergence between filterj and filterk
                
                # layeri_softmaxP = layeri_softmaxP.cuda()
                # layeri_KL = layeri_KL.cuda()

                #* compute layeri_iScore
                layeri_iScore = torch.sum(layeri_KL, dim=1)        #* layeri_iScore.shape=[cout], layeri_iScore[j] means filterj's importance score


                #* setup conv_module's traning-aware attr.
                ##* setup conv_module.epoch
                module.epoch = epoch

                ##* setup conv_module.layeri_topm_filters_id
                _, topm_ids = torch.topk(layeri_iScore, layers_m[layerid])
                # module.layeri_topm_filters_id = topm_ids

                ##* setup conv_module.layeri_softmaxP
                module.layeri_softmaxP = layeri_softmaxP[topm_ids, :]

                ###* printP
                bottom_num = layers_cout[layerid]-layers_m[layerid]
                if bottom_num > 0:
                    _, bottom_ids = torch.topk(layeri_iScore, bottom_num, largest=False)
                    with open(args.job_dir + '/softmaxP.log',"a") as f:
                        f.write(f'==================== Epoch:{epoch}, layer {layerid}, m:{len(topm_ids)} ==================== \
                                \n\nP(m*n):{torch.max(layeri_softmaxP[topm_ids, :], dim=1)}, \n\nP((n-m)*n):{torch.max(layeri_softmaxP[bottom_ids, :], dim=1)} \n\n\n')
            print(f'cost: {time.time()-start}')

            del param, layeri_param, layeri_negaEudist, layeri_KL, layeri_iScore, topm_ids

            train(model, optimizer, trainLoader, args, epoch, topk=(1, 5))
            test_top1_acc, test_top5_acc = test(model, testLoader, topk=(1, 5))
            
            is_best = best_top5_acc < test_top5_acc
            best_top1_acc = max(best_top1_acc, test_top1_acc)
            best_top5_acc = max(best_top5_acc, test_top5_acc)

            
            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            
            state = {
                'state_dict': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_top1_acc': best_top1_acc,
                'best_top5_acc': best_top5_acc,
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
        
        logger.info(f'Best Acc-top1: {float(best_top1_acc):.3f}, Acc-top5: {float(best_top5_acc):.3f}')


        #* Test compact model
        compact_model = compact_model.to(device)
        compact_state_dict = torch.load(f'{args.job_dir}/checkpoint/model_best_compact.pt')
        compact_model.load_state_dict(compact_state_dict)
        logger.info(f'Best Compact model accuracy:')
        compact_test_acc = float(test(compact_model, loader.testLoader))

if __name__ == '__main__':
    main()
