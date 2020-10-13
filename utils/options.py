import argparse

parser = argparse.ArgumentParser(description='Pytorch ImageNet Training')

#* dataset-related [data_set, data_path]
parser.add_argument(
    '--data_set',
    type=str,
    # default='imagenet',     # for imagenet
    default='cifar10',      # for cifar10
    help='Name of dataset. [cifar10, imagenet] default: '
)

parser.add_argument(
    '--data_path',
    type=str,
    default='/home/cbh/one/Dataset',  # for cifar10
    # default='/media/disk2/data/ImageNet2012', # for imagenet 14
    # default='/media/disk1/lishaojie/ImageNet2012', # for imagenet 22
    help='Path to dataset.'
)


#* training params [train_batch_size, test_batch_size,num_epoch, gpus]
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=256,
    help='Batch size of all GPUS.'
)

parser.add_argument(
    '--test_batch_size',
    type=int,
    default=256,
    help='Batch size of all GPUS.'
)

parser.add_argument(
    '--num_epochs',
    type=int,
    default=150,    # for cifar10
    # default=90,     # for imagenet
    help='Num of total epochs to run. default: 10'
)

parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='Select gpu_id to use. default: [0]',
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.1,   # for cifar10/imagenet
    help='Learning rate. default: 0.01'
)

parser.add_argument(
    '--lr_decay_step',
    type=int,
    nargs='+',
    default=[50, 100],    # for cifar10
    # default=None,           # for imagenet
    help='The iterval of learning rate decay (for cifar10). default: [50, 100]'
)

parser.add_argument(
    '--lr_type',
    default='step', 
    type=str, 
    help='lr scheduler (step/exp/cos/step3/fixed)'
)

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,    # for cifar10
    # default=0.99,     # for imagenet
    help='Momentum. default: 0.9'
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-4,   # for cifar10
    # default=1e-4,   # for imagenet
    help='Weight decay.'
)


#* model-related [arch, resume, test_only]
parser.add_argument(
    '--arch',
    type=str,
    default='vgg16',    # for cifar10
    # default='resnet50', # for imagenet
    help='Model architecture. default: resnet18'
)

parser.add_argument(
    '--resume',
    type=str,
    default=None,
    help='Path to checkpoint file. default: None'
)

parser.add_argument(
    '--test_only',
    action='store_true',
    default=False,
    help='Test model on testset. default: False'
)


#* file-related [job_dir, reset, rm_old_ckpt]
parser.add_argument(
    '--job_dir',
    type=str,
    default='/media/disk1/cbh/EXP/tmp',
    help='The directory where the summaries will be sotred. default: ./experiment',
)

parser.add_argument(
    '--reset',
    default=False,
    action='store_true',
    help='reset the directory?'
)

parser.add_argument(
    '--debug',
    default=False,
    action='store_true',
    help='input to open debug state'
)

#* pruning-related
parser.add_argument(
    '--cprate',
    type=str,
    default=None,
    help='compress rate of each conv.'
)



args = parser.parse_args()

