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


# init
device = torch.device(f'cuda:{args.gpus[0]}') if torch.cuda.is_available() else 'cpu'
checkpoint = utils.CheckPoint(args)
logger = utils.GetLogger(os.path.join(args.job_dir + '/logger.log'))
loss_func = nn.CrossEntropyLoss()


# Data
print('==> Preparing data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)

default_cprate={
    'vgg16':    [0.7]*7+[0.1]*6,
    'resnet56': [0.6]+[0.7]+[0.5]+[0.5]+[0.4]+[0.2]+[0.3]+[0.4]+[0.8]+
                [0.7]+[0.6]+[0.9]+[0.8]+[0.9]+[0.8]+[0.4]+[0.2]+[0.2]+
                [0.7]+[0.3]+[0.8]+[0.4]+[0.3]+[0.7]+[0.2]+[0.4]+[0.8]+
                [0.0]+[0.0]+[0.0],
    'resnet110':[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+[0.1]+
                [0.2]+[0.2]+[0.2]+[0.2]+[0.2]+[0.2]+[0.2]+[0.2]+[0.2]+[0.2]+[0.2]+[0.2]+[0.2]+[0.2]+[0.2]+[0.2]+[0.2]+[0.2]+
                [0.3]+[0.3]+[0.3]+[0.3]+[0.3]+[0.3]+[0.3]+[0.3]+[0.3]+[0.3]+[0.3]+[0.3]+[0.3]+[0.3]+[0.3]+[0.3]+[0.3]+[0.3]+
                [0.4]+[0.5]+[0.6],


    'resnet50': [0.2]+[0.8]*10+[0.8]*13+[0.55]*19+[0.45]*10,
}



#* compute cprate
if args.cprate:
    cprate = eval(args.cprate)
else:
    cprate = cprate = default_cprate[args.arch]

logger.info(f'cprate: \n{cprate}')


#* compute resnet lawer-wise cprate
resnet_block_num = {
    'resnet56': 9,
    'resnet110':18,
}
_cprate = []
if 'resnet' in args.arch:
    block_conv2_cprate = [val for val in cprate[-3:] for i in range(resnet_block_num[args.arch])]
    for item in zip(cprate[0:-3], block_conv2_cprate):
        _cprate += list(item)
    _cprate.insert(0, cprate[-3])
    cprate = _cprate

# Model
print('==> Building model..')
if args.arch == 'vgg16':
    model = Fused_VGG(vgg_name='vgg16', cprate=cprate)
    compact_model = Compact_VGG(vgg_name='vgg16', cprate=cprate)

elif args.arch == 'resnet56':
    model = fused_resnet56(cprate=cprate)
    compact_model = compact_resnet56(cprate=cprate)

elif args.arch == 'resnet110':
    model = fused_resnet110(cprate=cprate)
    compact_model = compact_resnet110(cprate=cprate)


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
    global model, compact_model

    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=args.momentum, weight_decay=args.weight_decay)
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
            # from collections import OrderedDict
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
    # Train from scratch
    else:
        start_epoch = 0
        best_acc = 0.0


    # test only
    if args.test_only:
        test(model, loader.testLoader)
        
    # train
    else:
        #* setup conv_modules
        conv_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_modules.append(module)


        #* setup conv_module.layerid / layers_cout
        layers_cout = []
        for layerid, module in enumerate(conv_modules):
            module.layerid = layerid
            layers_cout.append(module.out_channels)

        #* compute layers_m
        layers_cout = np.asarray(layers_cout)
        layers_cprate = np.asarray(cprate)
        layers_m = (layers_cout * (1-layers_cprate)).astype(int)



        for epoch in range(start_epoch, start_epoch + args.num_epochs):
            #* compute t, t tends to 0 as epochs increases to num_epochs.
            # t = 1 - epoch / args.num_epochs
            t=eval(args.t_expression)

            #* compute layeri_param / layeri_negaEudist / layeri_softmaxP / layeri_KL / layeri_iScore
            start = time.time()
            for layerid, module in enumerate(conv_modules):
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
                layeri_KL = torch.mean(layeri_softmaxP[:,None,:] * (layeri_softmaxP[:,None,:]/layeri_softmaxP).log(), dim = 2)      #* layeri_KL.shape=[cout, cout], layeri_KL[j, k] means KL divergence between filterj and filterk

                #* compute layeri_iScore
                layeri_iScore = torch.mul(torch.mean(layeri_KL, dim=1), -1)     #* layeri_iScore.shape=[cout], layeri_iScore[j] means filterj's importance score


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

            #* 
            compact_state_dict = OrderedDict()
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    compact_state_dict[name+'.weight'] = module.fused_weight
                    if module.bias is not None:
                        compact_state_dict[name+'.bias'] = module.fused_bias

                if isinstance(module, nn.BatchNorm2d):
                    compact_state_dict[name+'.weight'] = module.weight
                    compact_state_dict[name+'.bias'] = module.bias
                    compact_state_dict[name+'.running_mean'] = module.running_mean
                    compact_state_dict[name+'.running_var'] = module.running_var
                    compact_state_dict[name+'.num_batches_tracked'] = module.num_batches_tracked
                if isinstance(module, nn.Linear):
                    compact_state_dict[name+'.weight'] = module.weight
                    compact_state_dict[name+'.bias'] = module.bias
            
            compact_model.load_state_dict(compact_state_dict)

        compact_model = compact_model.to(device)
        logger.info(f'Compact model:')
        compact_test_acc = float(test(compact_model, loader.testLoader))


        logger.info(f'Best accuracy: {best_acc:.3f}')

if __name__ == '__main__':
    main()
