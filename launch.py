#!/usr/bin/env python

import os
import sys
import torch
from savanna.distributed import initialize_distributed
from savanna.utils.env import set_env
from savanna.utils.logging import init_wandb
from savanna.utils.options import get_args
from savanna.utils.checkpointing import load_checkpoint
from savanna.model.language_model import get_language_model
from savanna.model.criterion import get_loss_func
from savanna.data.dataset import build_train_valid_test_datasets
from savanna.optim import get_optimizer, get_learning_rate_scheduler
from savanna.trainer import Trainer

def main():
    # 获取参数
    args = get_args()
    
    # 设置环境
    set_env(args)
    
    # 初始化分布式训练
    initialize_distributed(args)
    
    # 初始化wandb（可选）
    if args.wandb_project and args.rank == 0:
        init_wandb(args)
    
    # 构建数据集
    train_data, valid_data, test_data = build_train_valid_test_datasets(args)
    
    # 构建模型
    model = get_language_model(args)
    
    # 加载检查点（如果有）
    iteration = load_checkpoint(args, model) or 0
    
    # 获取损失函数
    criterion = get_loss_func(args)
    
    # 获取优化器
    optimizer = get_optimizer(model, args)
    
    # 获取学习率调度器
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)
    
    # 创建训练器
    trainer = Trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data=train_data,
        valid_data=valid_data,
        criterion=criterion
    )
    
    # 开始训练
    trainer.train(iteration)

if __name__ == '__main__':
    main()