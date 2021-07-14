import argparse

parser = argparse.ArgumentParser(description='Pytorch ImageNet Training')



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
    default=None,
    help='Batch size of all GPUS.'
)

parser.add_argument(
    '--test_batch_size',
    type=int,
    default=None,
    help='Batch size of all GPUS.'
)

parser.add_argument(
    '--num_epochs',
    type=int,
    default=None,
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
    default=None,
    help='Learning rate.'
)

parser.add_argument(
    '--lr_decay_step',
    type=int,
    nargs='+',
    default=None,
    help='The iterval of learning rate decay (for cifar10). default: [50, 100]'
)

parser.add_argument(
    '--lr_type',
    default=None,
    type=str, 
    help='lr scheduler (step/exp/cos/step3/fixed)'
)

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum. default: 0.9'
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=None,
    help='Weight decay.'
)



parser.add_argument(
    '--arch',
    type=str,
    default=None,
    help='Model architecture. default: resnet18'
)

parser.add_argument(
    '--resume',
    action='store_true',
    default=False,
    help='resume . default: False'
)

parser.add_argument(
    '--test_only',
    action='store_true',
    default=False,
    help='Test model on testset. default: False'
)


parser.add_argument(
    '--job_dir',
    type=str,
    default=None,
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

parser.add_argument(
    '--get_flops',
    default=False,
    action='store_true',
    help='get_flops'
)



parser.add_argument(
    '--cprate',
    type=str,
    default=None,
    help='compress rate of each conv.'
)



args = parser.parse_args()

