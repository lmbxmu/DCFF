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


# init
device = torch.device(f'cuda:{args.gpus[0]}') if torch.cuda.is_available() else 'cpu'
checkpoint = utils.CheckPoint(args)
logger = utils.GetLogger(os.path.join(args.job_dir + '_logger.log'))
loss_func = nn.CrossEntropyLoss()


# Data
print('==> Preparing data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)


# Model
print('==> Building model..')
if args.arch == 'vgg11':
    model = VGG11()
elif args.arch == 'vgg13':
    model = VGG13()
elif args.arch == 'vgg16':
    model = VGG16()
elif args.arch == 'vgg19':
    model = VGG19()

elif args.arch == 'resnet18':
    model = ResNet18()
elif args.arch == 'resnet34':
    model = ResNet34()
elif args.arch == 'resnet50':
    model = ResNet50()
elif args.arch == 'resnet101':
    model = ResNet101()
elif args.arch == 'resnet152':
    model = ResNet152()

elif args.arch == 'googlenet':
    model = GoogLeNet()


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

    total = 0
    correct = 0


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
                f'Epoch[{epoch+1}] ({batch_idx * args.train_batch_size} / {len(trainLoader.dataset)}):\t'
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

#*
def kl(X):

    """
    Finds the pairwise Kullback-Leibler divergence
    matrix between all rows in X.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Array of probability data. Each row must sum to 1.

    Returns
    -------
    D : ndarray, shape (n_samples, n_samples)
        The Kullback-Leibler divergence matrix. A pairwise matrix D such that D_{i, j}
        is the divergence between the ith and jth vectors of the given matrix X.

    Notes
    -----
    Based on code from Gordon J. Berman et al.
    (https://github.com/gordonberman/MotionMapper)

    References:
    -----------
    Berman, G. J., Choi, D. M., Bialek, W., & Shaevitz, J. W. (2014). 
    Mapping the stereotyped behaviour of freely moving fruit flies. 
    Journal of The Royal Society Interface, 11(99), 20140672.
    """
    X = np.asarray(X, dtype=np.float)
    X_log = np.log(X)
    X_log[np.isinf(X_log) | np.isnan(X_log)] = 0

    entropies = -np.sum(X * X_log, axis=1)

    D = np.matmul(-X, X_log.T)
    D = D - entropies
    D = D / np.log(2)
    D *= (1 - np.eye(D.shape[0]))

    return D

default_cprate={
    'vgg16': [0.7]*7+[0.1]*6,
    'densenet40': [0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*5+[0.0],
    'googlenet': [0.10]+[0.7]+[0.5]+[0.8]*4+[0.5]+[0.6]*2,
    'resnet50':[0.2]+[0.8]*10+[0.8]*13+[0.55]*19+[0.45]*10,
    'resnet56':[0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4],
    'resnet110':[0.1]+[0.40]*36+[0.40]*36+[0.4]*36
}

# main function
def main():
    global model

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
        #* setup conv_modules / BN_modules
        conv_modules = []
        BN_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_modules.append(module)
            if isinstance(module, nn.BatchNorm2d):
                BN_modules.append(module)

        #* setup conv_module.layerid / layers_cout
        layers_cout = []
        for layerid, module in enumerate(conv_modules):
            module.layerid = layerid
            layers_cout.append(module.out_channels)

        #* compute layers_m
        layers_cout = np.asarray(layers_cout)
        layers_cprate = np.asarray(default_cprate[args.arch])
        layers_m = np.ceil(layers_cout * (1-layers_cprate)).astype(int)

        for epoch in range(start_epoch, start_epoch + args.num_epochs):
            #* compute t, t tends to 0 as epochs increases to num_epochs.
            t = 1 - epoch / args.num_epochs

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
                # layeri_KL = torch.from_numpy(kl(layeri_softmaxP.cpu())).to(device)

                #* compute layeri_iScore
                layeri_iScore = torch.sum(layeri_KL, dim=1)     #* layeri_iScore.shape=[cout], layeri_iScore[j] means filterj's importance score


                #* setup conv_module's traning-aware attr.
                ##* setup conv_module.epoch
                module.epoch = epoch

                ##* setup conv_module.layeri_topm_filters_id
                topm_value, topm_ids = torch.topk(layeri_iScore, layers_m[layerid])
                module.layeri_topm_filters_id = topm_ids

                ##* setup conv_module.layeri_softmaxP
                module.layeri_softmaxP = layeri_softmaxP

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
        
        logger.info(f'Best accuracy: {best_acc:.3f}')

if __name__ == '__main__':
    main()
