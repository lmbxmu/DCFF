# cifar/res56/ABCP

## 5e-4
python main_cifar.py \
--arch resnet56 \
--lr 0.1 \
--num_epochs 300 \
--lr_decay_step 150 225 \
--weight_decay 5e-4 \
--cprate '[0.6]+[0.7]+[0.5]+[0.5]+[0.4]+[0.2]+[0.3]+[0.4]+[0.8] + [0.7]+[0.6]+[0.9]+[0.8]+[0.9]+[0.8]+[0.4]+[0.2]+[0.2] + [0.7]+[0.3]+[0.8]+[0.4]+[0.3]+[0.7]+[0.2]+[0.4]+[0.8] + [0.0]+[0.0]+[0.0]' \
--job_dir '/media/disk1/cbh/EXP/2020/09/KL-score/cifar/res56/run-n/e-300/ABCP-cp1/5e-4/debug/debug-res56-lr0.1-[-dist+10^-7+t=0.5+Pupdate_all]' \
--gpus 1 \
--data_set 'cifar10' \
--data_path '/home/cbh/one/Dataset' \
--momentum 0.9 \
--reset