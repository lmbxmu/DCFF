# cifar/res56/ABCP

## 5e-4
python main_cifar.py \
--arch vgg16 \
--lr 0.1 \
--num_epochs 300 \
--lr_decay_step 150 225 \
--weight_decay 5e-4 \
--cprate '[0.0]*13' \
--job_dir '/media/disk1/cbh/EXP/2020/09/KL-score/cifar/res56/run-n/e-300/ABCP-cp1/5e-4/debug-[-dist+10^-7+t=1+Pupdate1]' \
--gpus 3 \
--reset