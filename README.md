# CVPR21

## CIFAR-10

### vgg16

#### epoch-300

```
2020-10-01-12:00:35

data_set: cifar10
data_path: /home/cbh/one/Dataset
train_batch_size: 256
test_batch_size: 256
num_epochs: 300
gpus: [0]
lr: 0.1
lr_decay_step: [150, 225]
momentum: 0.9
weight_decay: 0.0005
arch: vgg16
resume: None
test_only: False
job_dir: /media/disk1/cbh/EXP/2020/09/KL-score/cifar/vgg16/run-n/e-300/HRankP-cp3/r0
reset: False
cprate: [0.45]*7+[0.78]*5+[0.0]
t_expression: 1 - epoch / args.num_epochs
```



| run-n        | best top1 | FLOPs/M (PR)      | Params/M (PR)    |
| ------------ | --------- | ----------------- | ---------------- |
| **baseline** | **93.96** | **313.73 (0.0%)** | **14.98 (0.0%)** |
| 0            | 93.31     | 313.73 (0.0%)     | **14.98 (0.0%)** |
| 1            | 92.66     |                   |                  |
| 2            | 93.05     |                   |                  |
| 3            | 92.94     |                   |                  |
| 4            | **93.33** |                   |                  |
| 5            | 92.57     |                   |                  |
| 6            | 92.57     |                   |                  |
| 7            | 92.89     |                   |                  |
| 8            | 92.89     |                   |                  |
| 9            | 92.88     |                   |                  |
| 10           | 93.14     |                   |                  |
| 11           | 93.00     |                   |                  |
| 12           | 92.78     |                   |                  |
| 13           | 93.22     |                   |                  |
| 14           | 92.79     |                   |                  |
| 15           | 92.93     |                   |                  |
| 16           | 93.15     |                   |                  |
| 17           | 92.95     |                   |                  |
| 18           | 93.00     |                   |                  |
| 19           |           |                   |                  |
| **avg**      |           |                   |                  |

<br>

### resnet56

#### epoch-300

```
2020-10-01-17:50:26

data_set: cifar10
data_path: /home/cbh/one/Dataset
train_batch_size: 256
test_batch_size: 256
num_epochs: 300
gpus: [1]
lr: 0.1
lr_decay_step: [150, 225]
momentum: 0.9
weight_decay: 0.0005
arch: resnet56
resume: None
test_only: False
job_dir: /media/disk1/cbh/EXP/2020/09/KL-score/cifar/res56/run-n/e-300/ABCP-cp1/r0
reset: False
cprate: [0.6]+[0.7]+[0.5]+[0.5]+[0.4]+[0.2]+[0.3]+[0.4]+[0.8] + [0.7]+[0.6]+[0.9]+[0.8]+[0.9]+[0.8]+[0.4]+[0.2]+[0.2] + [0.7]+[0.3]+[0.8]+[0.4]+[0.3]+[0.7]+[0.2]+[0.4]+[0.8] + [0.0]+[0.0]+[0.0]
t_expression: 1 - epoch / args.num_epochs
```



| run-n        | best top1/% | FLOPs/M (PR)      | Params/M (PR)   |
| ------------ | ----------- | ----------------- | --------------- |
| **baseline** | **93.26**   | **125.49M(0.0%)** | **0.85M(0.0%)** |
| 0            | 92.23       |                   |                 |
| 1            | 92.68       |                   |                 |
| 2            | 91.59       |                   |                 |
| 3            | 92.82       |                   |                 |
| 4            |             |                   |                 |
| 5            |             |                   |                 |
| 6            |             |                   |                 |
| 7            |             |                   |                 |
| 8            |             |                   |                 |
| 9            |             |                   |                 |

### resnet110

```
2020-10-01-20:11:19

data_set: cifar10
data_path: /home/cbh/one/Dataset
train_batch_size: 256
test_batch_size: 256
num_epochs: 300
gpus: [3]
lr: 0.1
lr_decay_step: [150, 225]
momentum: 0.9
weight_decay: 0.0005
arch: resnet110
resume: None
test_only: False
job_dir: /media/disk1/cbh/EXP/2020/09/KL-score/cifar/res110/run-n/e-300/ABCP-cp1/r1
reset: False
cprate: [0.2]+[0.0]+[0.2]+[0.3]+[0.6]+[0.7]+[0.1]+[0.3]+[0.3]+[0.4]+[0.7]+[0.7]+[0.5]+[0.1]+[0.3]+[0.0]+[0.6]+[0.0] + [0.2]+[0.5]+[0.0]+[0.6]+[0.7]+[0.5]+[0.7]+[0.7]+[0.3]+[0.4]+[0.0]+[0.3]+[0.1]+[0.5]+[0.0]+[0.1]+[0.0]+[0.7] + [0.0]+[0.1]+[0.3]+[0.3]+[0.3]+[0.1]+[0.2]+[0.5]+[0.7]+[0.2]+[0.4]+[0.7]+[0.5]+[0.7]+[0.7]+[0.7]+[0.5]+[0.1] + [0.6]+[0.2]+[0.5]
t_expression: 1 - epoch / args.num_epochs
```



| run-n        | best top1/% | FLOPs/M (PR)      | Params/M (PR)   |
| ------------ | ----------- | ----------------- | --------------- |
| **baseline** | **93.50**   | **252.89M(0.0%)** | **1.72M(0.0%)** |
| 0            | 92.77       |                   |                 |
| 1            | 92.36       |                   |                 |
| 2            |             |                   |                 |
| 3            |             |                   |                 |
| 4            |             |                   |                 |
| 5            |             |                   |                 |
| 6            |             |                   |                 |
| 7            |             |                   |                 |
| 8            |             |                   |                 |
| 9            |             |                   |                 |





<br>

### MobileNetV1

<br>

## ImageNet

<br>
