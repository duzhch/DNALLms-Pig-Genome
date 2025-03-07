"""检查点相关工具函数"""

import os
import torch
import logging

def save_checkpoint(args, iteration, model, optimizer=None, lr_scheduler=None):
    """保存检查点
    
    Args:
        args: 配置参数
        iteration: 当前迭代次数
        model: 模型
        optimizer: 优化器
        lr_scheduler: 学习率调度器
    """
    if args.rank != 0:
        return
        
    checkpoint = {
        'iteration': iteration,
        'model': model.state_dict()
    }
    
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    
    if lr_scheduler is not None:
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
        
    checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{iteration}.pt')
    torch.save(checkpoint, checkpoint_path)
    logging.info(f'保存检查点到 {checkpoint_path}')
    
def load_checkpoint(args, model, optimizer=None, lr_scheduler=None):
    """加载检查点
    
    Args:
        args: 配置参数
        model: 模型
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        
    Returns:
        iteration: 加载的迭代次数
    """
    if not os.path.exists(args.checkpoint_dir):
        return None
        
    # 获取最新的检查点
    checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.startswith('checkpoint_')]
    if not checkpoints:
        return None
        
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
    checkpoint_path = os.path.join(args.checkpoint_dir, latest_checkpoint)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
    logging.info(f'加载检查点 {checkpoint_path}')
    return checkpoint['iteration']