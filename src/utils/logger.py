import logging
import os
import sys
from datetime import datetime

def setup_logger(log_dir="outputs/logs", name="quant_project"):
    """
    配置全局 Logger：同时输出到控制台和文件
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. 创建 Logger 对象
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 防止重复添加 Handler (如果多次调用 setup_logger)
    if not logger.handlers:
        # 2. 定义格式：[时间] [等级] [文件名:行号] - 信息
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 3. 创建控制台 Handler (输出到屏幕)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 4. 创建文件 Handler (保存到 outputs/logs/2024-xx-xx.log)
        log_filename = f"{datetime.now().strftime('%Y-%m-%d')}.log"
        file_handler = logging.FileHandler(
            os.path.join(log_dir, log_filename), encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# 直接初始化一个全局实例，方便其他地方直接 import logger
logger = setup_logger()