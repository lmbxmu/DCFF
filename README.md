# CVPR21

## CIFAR-10

### vgg16

#### epoch-150

```
data_set: cifar10
data_path: /home/cbh/one/Dataset
train_batch_size: 256
test_batch_size: 256
num_epochs: 150
gpus: [0]
lr: 0.1
lr_decay_step: [50, 100]
momentum: 0.9
weight_decay: 0.0005
arch: vgg16
resume: None
test_only: False
cprate: [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.78, 0.78, 0.78, 0.78, 0.78, 0.0]
```

| run-n   | bast top1 |
| ------- | --------- |
| 1       | 92.17     |
| 2       | 92.09     |
| 3       | 91.45     |
| 4       | 91.98     |
| 5       | 92.31     |
| 6       | 91.88     |
| 7       | 92.30     |
| 8       | 91.78     |
| 9       | 91.63     |
| 10      | **92.48** |
| 11      | 92.14     |
| 12      | 91.91     |
| 13      | 91.53     |
| 14      | 91.78     |
| 15      | 92.28     |
| 16      | 91.60     |
| 17      | 92.19     |
| 18      | 92.08     |
| 19      | 92.03     |
| 20      | 92.30     |
| **avg** | **92.00** |



#### epoch-300

```
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
arch: vgg16
resume: None
test_only: False
cprate: [0.45]*7+[0.78]*5+[0.0]
```



| run-n   | bast top1 |
| ------- | --------- |
| 1       | 92.78     |
| 2       | 92.69     |
| 3       | 92.85     |
| 4       | 92.90     |
| 5       | 92.76     |
| 6       | 92.75     |
| 7       | 92.67     |
| 8       | 92.97     |
| 9       | **93.05** |
| 10      | 92.68     |
| **avg** | 92.81     |



### resnet56



### MobileNet



## ImageNet

