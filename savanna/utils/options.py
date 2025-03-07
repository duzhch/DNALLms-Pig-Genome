"""参数解析相关工具函数"""

import os
import yaml
import argparse

def get_args():
    """获取配置参数
    
    Returns:
        args: 配置参数
    """
    parser = argparse.ArgumentParser(description='DNA语言模型训练')
    
    # 基础配置
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--debug', action='store_true', help='是否开启调试模式')
    
    # 分布式训练配置
    parser.add_argument('--master-addr', type=str, default='localhost', help='主节点地址')
    parser.add_argument('--master-port', type=int, default=29500, help='主节点端口')
    parser.add_argument('--distributed-backend', type=str, default='nccl', help='分布式后端')
    
    # wandb配置
    parser.add_argument('--wandb-project', type=str, default=None, help='wandb项目名称')
    parser.add_argument('--run-name', type=str, default=None, help='实验运行名称')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新参数
    for k, v in config.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                setattr(args, f'{k}_{sub_k}', sub_v)
        else:
            setattr(args, k, v)
    
    return args