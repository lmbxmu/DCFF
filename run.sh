# cifar/res56/ABCP

python main_cifar.py \
--data_set 'cifar10' \
--data_path '/home/cbh/one/Dataset' \
--train_batch_size 256 \
--test_batch_size 256 \
--num_epochs 300 \
--lr 0.1 \
--lr_decay_step 150 225 \
--lr_type 'step' \
--momentum 0.9 \
--weight_decay 5e-4 \
--arch vgg16 \
--cprate '[0.19]*2+[0.35]*2+[0.54]*3+[0.84]*3+[0.76]*3' \
--job_dir '/media/disk1/cbh/EXP/20.09/cvpr21/1013/vgg16_cp1_r0' \
--reset \
--gpus 0