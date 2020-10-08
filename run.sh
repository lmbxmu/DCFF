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
--arch resnet56 \
--cprate '[0.6]+[0.7]+[0.5]+[0.5]+[0.4]+[0.2]+[0.3]+[0.4]+[0.8]+ \
          [0.7]+[0.6]+[0.9]+[0.8]+[0.9]+[0.8]+[0.4]+[0.2]+[0.2]+ \
          [0.7]+[0.3]+[0.8]+[0.4]+[0.3]+[0.7]+[0.2]+[0.4]+[0.8]+ \
          [0.0]+[0.0]+[0.0]' \
--t_expression '0.1' \
--p_type 'softmax' \
--kl_add 1e-7 \
--compute_wd \
--job_dir '/media/disk1/cbh/EXP/20.09/cvpr21/1008/res56_t=0.1_p=softmax_kl-add=1e-7_compute-wd_r3' \
--reset \
--gpus 0