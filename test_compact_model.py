import argparse
import torch
import torch.nn as nn
from thop import profile
import time

import data.cifar10 as cifar10
import data.imagenet as imagenet
import utils.common as utils



parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_set',
    type=str,
    default=None,
    help='Name of dataset. [cifar10, imagenet] default: '
)

parser.add_argument(
    '--data_path',
    type=str,
    default=None,
    help='Path to dataset.'
)

parser.add_argument(
    '--train_batch_size',
    type=int,
    default=256,
    help='Batch size of all GPUS.'
)

parser.add_argument(
    '--test_batch_size',
    type=int,
    default=None,
    help='Batch size of all GPUS.'
)

parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='Select gpu_id to use. default: [0]',
)

parser.add_argument(
    '--arch',
    type=str,
    default=None,
    help='Model architecture. default: resnet18'
)

parser.add_argument(
    '--cprate',
    type=str,
    default=None,
    help='compress rate of each conv.'
)

parser.add_argument(
    '--resume_compact_model',
    type=str,
    default=None,
)

args = parser.parse_args()

device = torch.device(f'cuda:{args.gpus[0]}') if torch.cuda.is_available() else 'cpu'
loss_func = nn.CrossEntropyLoss()

if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
elif args.data_set == 'imagenet':
    loader = imagenet.Data(args)

#* Testing function
if args.data_set == 'cifar10':
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
            print(
                f'Test Loss: {float(losses.avg):.4f}\t Acc: {float(accuracy.avg):.2f}%\t\t Time: {(current_time - start_time):.2f}s'
            )
        return accuracy.avg
elif args.data_set == 'imagenet':
    def test(model, testLoader, topk=(1,)):
        model.eval()

        losses = utils.AverageMeter()
        top1_accuracy = utils.AverageMeter()
        top5_accuracy = utils.AverageMeter()

        start_time = time.time()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testLoader):
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
            print(
                f'Test Loss: {float(losses.avg):.6f}\t Top1: {float(top1_accuracy.avg):.6f}%\t'
                f'Top5: {float(top5_accuracy.avg):.6f}%\t Time: {float(current_time - start_time):.2f}s'
            )

        return float(top1_accuracy.avg), float(top5_accuracy.avg)

#* Compute resnet lawer-wise cprate
if args.cprate:
    cprate = eval(args.cprate)
else:
    cprate = default_cprate[args.arch]

print(f'{args.arch}\'s cprate: \n{cprate}')

if args.data_set == 'cifar10':
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

elif args.data_set == 'imagenet':
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
if args.data_set == 'cifar10':
    from models.cifar import *
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

elif args.data_set == 'imagenet':
    from models.imagenet import *
    if args.arch == 'resnet50':
        model = fused_resnet50(cprate=layer_wise_cprate)
        compact_model = compact_resnet50(cprate=layer_wise_cprate)
        origin_model = origin_resnet50()
    else:
        raise NotImplementedError
print(model)

#* Test compact model
compact_model = compact_model.to(device)
compact_state_dict = torch.load(f'{args.resume_compact_model}')
compact_model.load_state_dict(compact_state_dict)

if args.data_set == 'cifar10':
    compact_test_acc = float(test(compact_model, loader.testLoader))
    print(f'Best Compact Model Acc Top1: {compact_test_acc:.2f}%')
    inputs = torch.randn(1, 3, 32, 32)

elif args.data_set == 'imagenet':
    compact_test_acc = test(compact_model, loader.testLoader, topk=(1, 5))
    print(f'Best Compact Model Acc Top1: {compact_test_acc[0]:.2f}%, Top5: {compact_test_acc[1]:.2f}%')
    inputs = torch.randn(1, 3, 224, 224)



#* Compute flops, flops, puring rate
compact_model = compact_model.cpu()
origin_model = origin_model.cpu()
compact_flops, compact_params = profile(compact_model, inputs=(inputs, ))
origin_flops, origin_params = profile(origin_model, inputs=(inputs, ))

flops_prate = (origin_flops-compact_flops)/origin_flops
params_prate = (origin_params-compact_params)/origin_params
print(f'{args.arch}\'s baseline model: FLOPs = {origin_flops/10**6:.2f}M (0.0%), Params = {origin_params/10**6:.2f}M (0.0%)')
print(f'{args.arch}\'s pruned   model: FLOPs = {compact_flops/10**6:.2f}M ({flops_prate*100:.2f}%), Params = {compact_params/10**6:.2f}M ({params_prate*100:.2f}%)')
