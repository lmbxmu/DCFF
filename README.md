# Training Compact CNNs for Image Classification using Dynamic-coded Filter Fusion ![]( https://visitor-badge.glitch.me/badge?page_id=lmbxmu.dcff).


## Running Code

### Requirements

-  Pytorch >= 1.0.1
-  CUDA = 10.0.0
-  thop = 0.0.31

### Run Our Results

#### CIFAR-10

```shell
#* vgg16 step FLOPs_PR=76.8% Params_PR=92.8%
python main_cifar.py \
--data_set 'cifar10' \
--data_path 'DATASET' \
--train_batch_size 256 \
--test_batch_size 256 \
--num_epochs 300 \
--lr_decay_step 150 225 \
--lr 0.1 \
--weight_decay 5e-4 \
--lr_type 'step' \
--momentum 0.9 \
--arch vgg16 \
--cprate '[0.5]*2+[0.4]*2+[0.35]*3+[0.85]*6' \
--job_dir 'EXP' \
--gpus 0

#* googlenet step FLOPs_PR=70.1% Params_PR=66.3%
python main_cifar.py \
--data_set 'cifar10' \
--data_path 'DATASET' \
--train_batch_size 256 \
--test_batch_size 256 \
--num_epochs 300 \
--lr_decay_step 150 225 \
--lr 0.1 \
--weight_decay 5e-4 \
--lr_type 'step' \
--momentum 0.9 \
--arch googlenet \
--cprate '[0.0]*2+[0.8]+[0.0]+[0.8]*2+[0.0]*2+ ([0.0]+[0.9]+[0.0]+[0.9]*2+[0.0]*2)*3+ ([0.0]+[0.8]+[0.0]+[0.8]*2+[0.0]*2)*3+ ([0.0]+[0.9]+[0.0]+[0.9]*2+[0.0]*2)*2' \
--job_dir 'EXP' \
--gpus 0

#* resnet56 step FLOPs_PR=55.9% Params_PR=55.0%
python main_cifar.py \
--data_set 'cifar10' \
--data_path 'DATASET' \
--train_batch_size 256 \
--test_batch_size 256 \
--num_epochs 300 \
--lr_decay_step 150 225 \
--lr 0.1 \
--weight_decay 5e-4 \
--lr_type 'step' \
--momentum 0.9 \
--arch resnet56 \
--cprate '[0.7]*2+[0.5]*3+[0.3]*2+[0.4]+[0.8]+ [0.7]*2+[0.8]*4+[0.4]+[0.2]*2+[0.7]+[0.3]+[0.8]+[0.4]*2+[0.7]+[0.3]+[0.4]+[0.8]+ [0.0]*3' \
--job_dir 'EXP' \
--gpus 0

#* resnet110 step FLOPs_PR=66.6% Params_PR=67.9%
python main_cifar.py \
--data_set 'cifar10' \
--data_path 'DATASET' \
--train_batch_size 256 \
--test_batch_size 256 \
--num_epochs 300 \
--lr_decay_step 150 225 \
--lr 0.1 \
--weight_decay 5e-4 \
--lr_type 'step' \
--momentum 0.9 \
--arch resnet110 \
--cprate '[0.2]+[0.0]+[0.2]+[0.3]+[0.7]*2+[0.1]+[0.3]*2+[0.4]+[0.7]*2+[0.5]+[0.1]+[0.3]+[0.0]+[0.6]+[0.0]+[0.2]+[0.5]+[0.0]+[0.7]*2+[0.5]+[0.7]*2+[0.4]*2+[0.0]+[0.3]+[0.1]+[0.5]+[0.1]*3+[0.7]+ [0.1]*2+[0.3]*5+[0.5]+[0.7]+[0.2]+[0.4]+[0.7]*5+[0.5]+[0.1]+ [0.6]+[0.2]+[0.5]' \
--job_dir 'EXP' \
--gpus 0
```

#### ImageNet

```bash
#* resnet50 step FLOPs_PR=76.7% Params_PR=71.0%
python main_imagenet.py \
--data_set 'imagenet' \
--data_path 'DATASET' \
--train_batch_size 256 \
--test_batch_size 256 \
--num_epochs 90 \
--lr_type 'step' \
--lr 0.1 \
--weight_decay 1e-4 \
--momentum 0.9 \
--arch resnet50 \
--cprate '[0.0]+[0.8]*10+[0.7]*6+[0.6]*10+[0.4]*6+[0.3]*4' \
--job_dir 'EXP' \
--gpus 0

#* resnet50 step FLOPs_PR=63.8% Params_PR=58.6%
python main_imagenet.py \
--data_set 'imagenet' \
--data_path 'DATASET' \
--train_batch_size 256 \
--test_batch_size 256 \
--num_epochs 90 \
--lr_type 'step' \
--lr 0.1 \
--weight_decay 1e-4 \
--momentum 0.9 \
--arch resnet50 \
--cprate '[0.0]+[0.6]*10+[0.5]*6+[0.5]*10+[0.4]*6+[0.2]*4' \
--job_dir 'EXP' \
--gpus 0

#* resnet50 step FLOPs_PR=45.3% Params_PR=40.7%
python main_imagenet.py \
--data_set 'imagenet' \
--data_path 'DATASET' \
--train_batch_size 256 \
--test_batch_size 256 \
--num_epochs 90 \
--lr_type 'step' \
--lr 0.1 \
--weight_decay 1e-4 \
--momentum 0.9 \
--arch resnet50 \
--cprate '[0.0]+[0.35]*10+[0.3]*6+[0.4]*10+[0.3]*6+[0.1]*4' \
--job_dir 'EXP' \
--gpus 0

#* resnet50 cos FLOPs_PR=63.0% Params_PR=56.8%
python main_imagenet.py \
--data_set 'imagenet' \
--data_path 'DATASET' \
--train_batch_size 256 \
--test_batch_size 256 \
--num_epochs 180 \
--lr_type 'cos' \
--lr 0.01 \
--weight_decay 1e-4 \
--momentum 0.99 \
--arch resnet50 \
--cprate '[0.0]+([0.5]*10+[0.5]*6)*2+[0.25]*3+[0.0]' \
--job_dir 'EXP' \
--gpus 0


#* resnet50 cos FLOPs_PR=66.7% Params_PR=53.8%
python main_imagenet.py \
--data_set 'imagenet' \
--data_path 'DATASET' \
--train_batch_size 256 \
--test_batch_size 256 \
--num_epochs 180 \
--lr_type 'cos' \
--lr 0.01 \
--weight_decay 1e-4 \
--momentum 0.99 \
--arch resnet50 \
--cprate '[0.0]+([0.7]*7+[0.45]*9)*2+[0.24]*3+[0.0]' \
--job_dir 'EXP' \
--gpus 0

#* resnet50 cos FLOPs_PR=75.1% Params_PR=74.3%
python main_imagenet.py \
--data_set 'imagenet' \
--data_path 'DATASET' \
--train_batch_size 256 \
--test_batch_size 256 \
--num_epochs 180 \
--lr_type 'cos' \
--lr 0.01 \
--weight_decay 1e-4 \
--momentum 0.99 \
--arch resnet50 \
--cprate '[0.0]+([0.43]*7+[0.73]*9)*2+[0.45]*3+[0.0]' \
--job_dir 'EXP' \
--gpus 0
```



### Test our result

#### CIFAR-10

```bash
python test_compact_model.py \
--data_set 'cifar10' \ 
--data_path 'DATASET' \ # Input your data path of CIFAR-10 here
--test_batch_size 256 \
--arch [arch_name] \	# Input the corresponding network architecture here (vgg16/resnet56/resnet110/googlenet)
--cprate [cprate] \		# It can be found from the links in the following table
--resume_compact_model model_best_compact.pt \ # Input the pruned model path here. It can be downloaded from the links in the following table.
--gpus 0
```

| Full Model | Flops(PR)        | Parameter(PR)  | lr_type | Accuracy | Model                                                        |
| ---------- | ---------------- | -------------- | ------- | -------- | ------------------------------------------------------------ |
| VGG-16     | 72.77M (76.83%)  | 1.06M (92.80%) | step    | 93.47%   | [pruned](https://drive.google.com/drive/folders/12tTJ6xWPU_R4423ZSKrGbXd1plW8cQce?usp=sharing) |
| ResNet-56  | 55.84M (55.88%)  | 0.38M (54.95%) | step    | 93.26%   | [pruned](https://drive.google.com/drive/folders/16InC60b9tYdTGDQ-yU20AwbpISQQP89y?usp=sharing) |
| ResNet-110 | 85.30M (66.55%)  | 0.56M (67.86%) | step    | 93.80%   | [pruned](https://drive.google.com/drive/folders/1R5WDix2WyWD2cMF209j3zSohM_hSz2c_?usp=sharing) |
| GoogLeNet  | 457.22M (70.11%) | 2.08M (66.28%) | step    | 94.92%   | [pruned](https://drive.google.com/drive/folders/1Zk7IqvKuR6CkPPa6KLHY8b7P57tCcSkt?usp=sharing) |



#### ImageNet

```bash
python test_compact_model.py \
--data_set 'imagenet' \
--data_path 'DATASET' \	# Input your data path of ImageNet here
--test_batch_size 256 \	
--arch [arch_name] \	# Input the corresponding network architecture here (resnet50)
--cprate [cprate] \		# It can be found from the links in the following table
--resume_compact_model model_best_compact.pt \	# Input the pruned model path here. It can be downloaded from the links in the following table.
--gpus 0
```

| Full Model  | Flops(PR)      | Parameter(PR)   | lr_type | Top1-Accuracy | Top5- Accuracy | Model                                                        |
| ----------- | -------------- | --------------- | ------- | ------------- | -------------- | ------------------------------------------------------------ |
| ResNet-50   | 0.96B (76.70%) | 7.40M (71.03%)  | step    | 71.54%        | 90.57%         | [pruned](https://drive.google.com/drive/folders/1Zf0fsfAsgi0jB1ffjAXibzGAeo2MUCSg?usp=sharing) |
| ResNet-50   | 1.49B (63.75%) | 10.58M (58.60%) | step    | 74.21%        | 91.93%         | [pruned](https://drive.google.com/drive/folders/1E4QjP53_X8epM7NgwLQU3MH3SCV82cPx?usp=sharing) |
| ResNet-50   | 2.25B (45.30%) | 15.16M (40.67%) | step    | 75.18%        | 92.56%         | [pruned](https://drive.google.com/drive/folders/11OZ1-6h3XRad5shhR5zGNN9yUpIaDvjN?usp=sharing) |
| ResNet-50   | 1.52B (62.96%) | 11.05M (56.77%) | cos     | 75.60%        | 92.55%         | [pruned](https://drive.google.com/drive/folders/1lp0cY_5n1ZBZAFQQcJEewFXNItkqYftI?usp=sharing) |
| ResNet-50   | 1.38B (66.41%) | 11.81M (53.77%) | cos     | 74.85%        | 92.41%         | [pruned](https://drive.google.com/drive/folders/1KnTz3vVmy66iBWEqUcERgpCaSzZfDikx?usp=sharing) |
| ResNet-50   | 1.02B (75.11%) | 6.56M (74.33%)  | cos     | 73.81%        | 91.59%         | [pruned](https://drive.google.com/drive/folders/1GIDj-QuWeLL-_U1BTXqWmbOLRoGNErVf?usp=sharing) |
| MobileNetV2 | 140M(53.3%)    | 2.62M(25.1%)    | cos     | 68.60%        | 88.13%         | [pruned](https://drive.google.com/drive/folders/1OhnLFJVT_WPCSyL2z1NGpssf8cAnJQE3?usp=share_link) |



| Model               | Top1-Accuracy | Flops | Parameter | Link                                                         |
| ------------------- | ------------- | ----- | --------- | ------------------------------------------------------------ |
| DCFF $_{manually}$  | 71.54%        | 960M  | 7.40M     | [pruned](https://drive.google.com/drive/folders/1Zf0fsfAsgi0jB1ffjAXibzGAeo2MUCSg?usp=sharing) |
| DCFF $_{ABCPruner}$ | 72.19%        | 945M  | 7.35M     | [pruned](https://drive.google.com/drive/folders/1aYK-VMTFAWgxqB-tIcfl5pTbH_tnxlOV?usp=share_link) |
| DCFF $_{manually}$  | 74.21%        | 1490M | 10.58M    | [pruned](https://drive.google.com/drive/folders/1E4QjP53_X8epM7NgwLQU3MH3SCV82cPx?usp=sharing) |
| DCFF $_{ABCPruner}$ | 73.78%        | 1295M | 9.10M     | [pruned](https://drive.google.com/drive/folders/1KSzb7NHFkAhOTrhhognjl3rYldfp3cbY?usp=share_link) |
| DCFF $_{manually}$  | 74.21%        | 1490M | 10.58M    | [pruned](https://drive.google.com/drive/folders/1E4QjP53_X8epM7NgwLQU3MH3SCV82cPx?usp=sharing) |
| DCFF $_{ABCPruner}$ | 74.73%        | 1794M | 11.24M    | [pruned](https://drive.google.com/drive/folders/1eDiCkKbqRpTy6s6lvyKyxxAx7SIVu8kl?usp=share_link) |
| DCFF $_{manually}$  | 75.18%        | 2250M | 15.16M    | [pruned](https://drive.google.com/drive/folders/11OZ1-6h3XRad5shhR5zGNN9yUpIaDvjN?usp=sharing) |
| DCFF $_{ABCPruner}$ | 74.83%        | 1891M | 11.75M    | [pruned](https://drive.google.com/drive/folders/1qpz0n1uE-mQ4PtgjJbEWGbYJ3gzq6oN7?usp=share_link) |
| DCFF $_{manually}$  | 75.18%        | 2250M | 15.16M    | [pruned](https://drive.google.com/drive/folders/11OZ1-6h3XRad5shhR5zGNN9yUpIaDvjN?usp=sharing) |
| DCFF $_{ABCPruner}$ | 75.79%        | 2556M | 18.02M    | [pruned](https://drive.google.com/drive/folders/1a7i29gd4p8s7nEH3lKGPT36BwPh_Gh0C?usp=share_link) |
| DCFF $_{manually}$  | 71.54%        | 960M  | 7.40M     | [pruned](https://drive.google.com/drive/folders/1Zf0fsfAsgi0jB1ffjAXibzGAeo2MUCSg?usp=sharing) |
| DCFF $_{EagleEye}$  | 73.18%        | 1040M | 6.99M     | [pruned](https://drive.google.com/drive/folders/1E8MDiERJyUbXd6adWc-aQXKNA5YXrov2?usp=share_link) |
| DCFF $_{manually}$  | 75.18%        | 2250M | 15.16M    | [pruned](https://drive.google.com/drive/folders/11OZ1-6h3XRad5shhR5zGNN9yUpIaDvjN?usp=sharing) |
| DCFF $_{EagleEye}$  | 75.60%        | 2070M | 14.41M    | [pruned](https://drive.google.com/drive/folders/104tyqYWBb7TH3qAu4xP4-bkGf8KUTnyv?usp=share_link) |

