"""分布式训练工具"""

import os
import torch
import torch.distributed as dist

def initialize_distributed(args):
    """初始化分布式训练环境"""
    # 设置设备
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取分布式训练参数
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '1'))
    args.local_rank = int(os.getenv('LOCAL_RANK', '0'))
    
    if args.world_size > 1:
        # 初始化进程组
        dist.init_process_group(
            backend=args.distributed_backend,
            init_method='env://'
        )
        
        # 设置当前设备
        torch.cuda.set_device(args.local_rank)
        
        # 同步随机数生成器
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    return args

def cleanup():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """判断是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_world_size():
    """获取总进程数"""
    return dist.get_world_size() if dist.is_initialized() else 1

def get_rank():
    """获取当前进程序号"""
    return dist.get_rank() if dist.is_initialized() else 0

def all_reduce(tensor, op=dist.ReduceOp.SUM):
    """执行all-reduce操作"""
    if dist.is_initialized():
        dist.all_reduce(tensor, op)
    return tensor

def all_gather(tensor):
    """执行all-gather操作"""
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list