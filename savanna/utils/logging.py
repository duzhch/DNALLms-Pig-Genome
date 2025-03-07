"""日志和wandb相关工具函数"""

import os
import logging
import wandb

def init_logging(args):
    """初始化日志配置
    
    Args:
        args: 配置参数
    """
    # 设置日志级别
    log_level = os.getenv('SAVANNA_LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
def init_wandb(args):
    """初始化wandb
    
    Args:
        args: 配置参数
    """
    if not args.wandb_project:
        return
        
    # 设置wandb配置
    config = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'model_type': args.model_type,
        'max_seq_length': args.max_seq_length,
    }
    
    # 初始化wandb
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=config
    )
    
def log_metrics(metrics, step=None):
    """记录指标到wandb
    
    Args:
        metrics: 指标字典
        step: 当前步数
    """
    if wandb.run is not None:
        wandb.log(metrics, step=step)