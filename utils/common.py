from __future__ import absolute_import
import sys
import os
import datetime
import time
import shutil
import logging
import torch

from pathlib import Path


# 计算和保存当前参数的值
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    # 初始化参数
    def reset(self):
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  #计数

    # 更新参数
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count



# Save model and record configuration
class CheckPoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.args = args
        self.job_dir = Path(args.job_dir)
        self.ckpt_dir = self.job_dir / 'checkpoint'  
        self.run_dir = self.job_dir / 'run'

        if args.reset:
            print('rm -rf ' + args.job_dir)
            os.system('rm -rf ' + args.job_dir)
        
        if args.rm_old_ckpt:
            os.chdir(str(self.ckpt_dir))
            os.system('ls | grep [0-9] | xargs rm')
            os.chdir('../..')
            
        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)
        _make_dir(self.ckpt_dir)
        _make_dir(self.run_dir)

        config_dir = self.job_dir / 'config.txt'
        with open(config_dir, 'w') as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
            
    def save_model(self, state, epoch, is_best):
        save_path = f'{self.ckpt_dir}/model_{epoch:03d}.pt'
        torch.save(state, save_path)
        print('Save checkpoint.')
        if is_best:
            shutil.copyfile(save_path, f'{self.ckpt_dir}/model_best.pt')
            print('Save best model.')

def GetLogger(file_path):
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

# 计算acc，top-1和top-k
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

