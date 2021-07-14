import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

import data.cifar10 as cifar10
import utils.common as utils

from models.cifar import *

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
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
else:
    raise NotImplementedError


default_cprate={
    'vgg16':    [0.1]*13,
    'resnet56': [0.1]*30,
    'resnet110':[0.1]*57,
    'googlenet':[0.1]*64,
}

#* Compute cprate
if args.cprate:
    cprate = eval(args.cprate)
else:
    cprate = default_cprate[args.arch]

logger.info(f'{args.arch}\'s cprate: \n{cprate}')


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
    layer_wise_cprate = cprate
else:
    raise NotImplementedError
print(f'layer-wise cprate: \n{layer_wise_cprate}')


#* Model
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

#* Compute flops, flops, puring rate
if args.get_flops:
    inputs = torch.randn(1, 3, 32, 32)
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


#* Training function
def train(model, optimizer, trainLoader, args, epoch):
    print(f'Epoch: {epoch}')
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

        pred = utils.accuracy(outputs, targets)
        accuracy.update(pred[0], inputs.size(0))

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


#* Testing function
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
            pred = utils.accuracy(outputs, targets)
            accuracy.update(pred[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            f'Test Loss: {float(losses.avg):.4f}\t Acc: {float(accuracy.avg):.2f}%\t\t Time: {(current_time - start_time):.2f}s'
        )
    return accuracy.avg

#* main function
def main():
    global model, compact_model, origin_model

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

    #* Resume from checkpoint (Train from pre-train model)
    if args.resume:
        #* Load ckpt file.
        args.resume = args.job_dir+'/checkpoint/model_last.pt'        
        print(f'==> Loading cheakpoint {args.resume}......')
        assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'

        resume_ckpt = torch.load(args.resume, map_location=device)

        optimizer.load_state_dict(resume_ckpt['optimizer'])
        scheduler.load_state_dict(resume_ckpt['scheduler'])
        start_epoch = resume_ckpt['epoch'] + 1
        best_acc = resume_ckpt['best_acc']
        layers_iScore_resume = resume_ckpt['layers_iScore']

        tmp_state_dict = resume_ckpt['state_dict']
        
        if len(args.gpus) > 1:
            new_state_dict = OrderedDict()
            for k, v in tmp_state_dict.items():
                new_state_dict['module.' + k.replace('module.', '')] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(tmp_state_dict)


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
                layeri_KL = torch.mean(layeri_softmaxP[:,None,:] * (layeri_softmaxP[:,None,:]/(layeri_softmaxP+1e-7)).log(), dim = 2)      #* layeri_KL.shape=[cout, cout], layeri_KL[j, k] means KL divergence between filterj and filterk

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

            train(model, optimizer, loader.trainLoader, args, epoch)
            scheduler.step()
            test_acc = float(test(model, loader.testLoader))

            is_best = best_acc < test_acc
            best_acc = max(best_acc, test_acc)
            logger.info(f'Best Acc: {best_acc:.2f}%')


            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            
            state = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'state_dict': model_state_dict,
                'layers_iScore' : layers_iScore,
            }
            checkpoint.save_model(state, epoch, is_best)

            if is_best:
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

        logger.info(f'Best acc Top1: {best_acc:.2f}')

        #* Test compact model
        compact_model = compact_model.to(device)
        compact_state_dict = torch.load(f'{args.job_dir}/checkpoint/model_best_compact.pt')
        compact_model.load_state_dict(compact_state_dict)
        
        compact_test_acc = float(test(compact_model, loader.testLoader))
        logger.info(f'Best Compact Model Acc Top1: {compact_test_acc:.2f}%')

        #* Compute flops, flops, puring rate
        inputs = torch.randn(1, 3, 32, 32)
        compact_model = compact_model.cpu()
        origin_model = origin_model.cpu()
        compact_flops, compact_params = profile(compact_model, inputs=(inputs, ))
        origin_flops, origin_params = profile(origin_model, inputs=(inputs, ))

        flops_prate = (origin_flops-compact_flops)/origin_flops
        params_prate = (origin_params-compact_params)/origin_params
        logger.info(f'{args.arch}\'s baseline model: FLOPs = {origin_flops/10**6:.2f}M (0.0%), Params = {origin_params/10**6:.2f}M (0.0%)')
        logger.info(f'{args.arch}\'s pruned   model: FLOPs = {compact_flops/10**6:.2f}M ({flops_prate*100:.2f}%), Params = {compact_params/10**6:.2f}M ({params_prate*100:.2f}%)')


if __name__ == '__main__':
    main()
