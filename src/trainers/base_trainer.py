import os
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from src.utils.logger import logger  # 假设你已经有了 logger

class BaseTrainer(ABC):
    def __init__(self, model, config, device=None):
        """
        :param model: 初始化的模型对象
        :param config: 配置字典或对象 (来自你的 config.py)
        :param device: torch.device ('cuda' 或 'cpu')
        """
        self.model = model
        self.config = config
        
        # 1. 自动识别设备
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.model:
            self.model.to(self.device)
        
        # 2. 自动创建输出目录
        self.checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(f"Trainer 初始化完成。使用设备: {self.device}")
        logger.info(f"模型权重将保存在: {self.checkpoint_dir}")

    @abstractmethod
    def train(self, train_loader, val_loader=None):
        """
        子类必须实现具体的训练循环
        """
        pass

    def save_checkpoint(self, epoch, loss, filename="checkpoint.pth"):
        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}_{filename}")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'config': self.config
        }
        torch.save(checkpoint, path)
        logger.info(f"💾 Checkpoint 已保存: {path}")

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            logger.error(f"找不到权重文件: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ 成功加载权重: {path} (Epoch: {checkpoint.get('epoch')})")
        return checkpoint