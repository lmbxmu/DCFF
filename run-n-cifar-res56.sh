# cifar/res56/ABCP

python main_cifar.py \
--arch resnet56 \
--lr 0.1 \
--num_epochs 300 \
--lr_decay_step 150 225 \
--weight_decay 5e-4 \
--momentum 0.9 \
--data_set 'cifar10' \
--train_batch_size 256 \
--test_batch_size 256 \
--data_path '/home/cbh/one/Dataset' \
--cprate '[0.6]+[0.7]+[0.5]+[0.5]+[0.4]+[0.2]+[0.3]+[0.4]+[0.8] + [0.7]+[0.6]+[0.9]+[0.8]+[0.9]+[0.8]+[0.4]+[0.2]+[0.2] + [0.7]+[0.3]+[0.8]+[0.4]+[0.3]+[0.7]+[0.2]+[0.4]+[0.8] + [0.0]+[0.0]+[0.0]' \
--job_dir '/media/disk1/cbh/EXP/2020/09/KL-score/cifar/res56/run-n/e-300/ABCP-cp1/5e-4/topm_order/1006-t=[1-div[e,E]]+softmaxP' \
--t_expression '1- epoch / args.num_epochs' \
--reset \
--gpus 0

python main_cifar.py \
--arch resnet56 \
--lr 0.1 \
--num_epochs 300 \
--lr_decay_step 150 225 \
--weight_decay 5e-4 \
--momentum 0.9 \
--data_set 'cifar10' \
--train_batch_size 256 \
--test_batch_size 256 \
--data_path '/home/cbh/one/Dataset' \
--cprate '[0.6]+[0.7]+[0.5]+[0.5]+[0.4]+[0.2]+[0.3]+[0.4]+[0.8] + [0.7]+[0.6]+[0.9]+[0.8]+[0.9]+[0.8]+[0.4]+[0.2]+[0.2] + [0.7]+[0.3]+[0.8]+[0.4]+[0.3]+[0.7]+[0.2]+[0.4]+[0.8] + [0.0]+[0.0]+[0.0]' \
--job_dir '/media/disk1/cbh/EXP/2020/09/KL-score/cifar/res56/run-n/e-300/ABCP-cp1/5e-4/topm_order/1006-t=div[e+1,E]+softmaxP' \
--t_expression '(epoch+1) / args.num_epochs' \
--reset \
--gpus 0