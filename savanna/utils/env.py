"""环境配置相关工具函数"""

def set_env(args):
    """设置训练环境
    
    Args:
        args: 配置参数
    """
    import os
    import random
    import numpy as np
    import torch
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设置CUDA相关环境
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 设置CUDA内存分配器
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512'
    
    # 设置多进程启动方式
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    
    # 设置日志级别
    if args.debug:
        os.environ['SAVANNA_LOG_LEVEL'] = 'DEBUG'
    else:
        os.environ['SAVANNA_LOG_LEVEL'] = 'INFO'