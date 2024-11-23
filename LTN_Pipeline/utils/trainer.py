import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Union
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
import time
import json
from datetime import datetime

class Trainer:
    def __init__(
        self,
        yolo_extractor,  # YOLO特征提取器
        ltn_model,       # LTN网络
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda',
        save_dir: str = 'runs/',
        exp_name: Optional[str] = None
    ):
        self.device = device
        
        # 模型
        self.yolo_extractor = yolo_extractor.to(self.device)
        self.ltn_model = ltn_model.to(self.device)
        
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 优化器
        self.optimizer = optimizer or torch.optim.Adam(
            self.ltn_model.parameters(), 
            lr=0.001
        )
        
        # 设置保存目录
        self.save_dir = Path(save_dir)
        if exp_name is None:
            exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.save_dir / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 训练状态
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_satisfaction': [],
            'val_satisfaction': []
        }
        
        # 保存配置
        self._save_config()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('LTNTrainer')
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        fh = logging.FileHandler(self.exp_dir / 'train.log')
        fh.setLevel(logging.INFO)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    def _save_config(self):
        """保存训练配置"""
        config = {
            'device': self.device,
            'save_dir': str(self.save_dir),
            'exp_name': self.exp_dir.name,
            'optimizer': self.optimizer.__class__.__name__,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'batch_size': self.train_loader.batch_size,
            'num_workers': self.train_loader.num_workers
        }
        
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
            
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        **kwargs
    ):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'ltn_model_state': self.ltn_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            **kwargs
        }
        
        # 保存最新检查点
        torch.save(
            checkpoint,
            self.exp_dir / f'checkpoint_epoch_{epoch}.pt'
        )
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            torch.save(
                checkpoint,
                self.exp_dir / 'best_model.pt'
            )
            
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)
        
        self.ltn_model.load_state_dict(checkpoint['ltn_model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epoch = checkpoint['epoch']
        self.history = checkpoint['history']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch}")
        
    def train_epoch(self) -> dict:
        """训练一个epoch"""
        self.ltn_model.train()
        epoch_loss = 0
        epoch_satisfaction = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            
            # 提取YOLO特征
            with torch.no_grad():
                yolo_features = self.yolo_extractor.extract_features(images)
                
            # LTN前向传播
            self.optimizer.zero_grad()
            outputs = self.ltn_model(yolo_features)
            loss = self.ltn_model.compute_loss(outputs)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 更新统计
            epoch_loss += loss.item()
            epoch_satisfaction += outputs['satisfaction'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'satisfaction': f"{outputs['satisfaction'].item():.4f}"
            })
            
        # 计算平均值
        metrics = {
            'loss': epoch_loss / num_batches,
            'satisfaction': epoch_satisfaction / num_batches
        }
        
        return metrics
        
    @torch.no_grad()
    def validate(self) -> dict:
        """验证模型"""
        if self.val_loader is None:
            return {}
            
        self.ltn_model.eval()
        val_loss = 0
        val_satisfaction = 0
        num_batches = len(self.val_loader)
        
        pbar = tqdm(self.val_loader, desc='Validation')
        for images, labels in pbar:
            images = images.to(self.device)
            
            # 提取YOLO特征
            yolo_features = self.yolo_extractor.extract_features(images)
            
            # LTN前向传播
            outputs = self.ltn_model(yolo_features)
            loss = self.ltn_model.compute_loss(outputs)
            
            # 更新统计
            val_loss += loss.item()
            val_satisfaction += outputs['satisfaction'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'val_loss': f"{loss.item():.4f}",
                'val_satisfaction': f"{outputs['satisfaction'].item():.4f}"
            })
            
        # 计算平均值
        metrics = {
            'val_loss': val_loss / num_batches,
            'val_satisfaction': val_satisfaction / num_batches
        }
        
        return metrics
        
    def train(
        self,
        num_epochs: int,
        save_freq: int = 5,
        eval_freq: int = 1
    ):
        """完整的训练循环"""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(self.epoch, self.epoch + num_epochs):
            self.epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_satisfaction'].append(train_metrics['satisfaction'])
            
            # 验证
            if self.val_loader is not None and (epoch + 1) % eval_freq == 0:
                val_metrics = self.validate()
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['val_satisfaction'].append(val_metrics['val_satisfaction'])
                
                # 检查是否是最佳模型
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                
                # 记录日志
                self.logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Loss: {train_metrics['loss']:.4f} - "
                    f"Satisfaction: {train_metrics['satisfaction']:.4f} - "
                    f"Val Loss: {val_metrics['val_loss']:.4f} - "
                    f"Val Satisfaction: {val_metrics['val_satisfaction']:.4f}"
                )
            else:
                is_best = False
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.epoch + num_epochs} - "
                    f"Loss: {train_metrics['loss']:.4f} - "
                    f"Satisfaction: {train_metrics['satisfaction']:.4f}"
                )
            
            # 保存检查点
            if (epoch + 1) % save_freq == 0 or is_best:
                self.save_checkpoint(
                    epoch + 1,
                    is_best=is_best,
                    metrics={
                        'train': train_metrics,
                        'val': val_metrics if self.val_loader is not None else {}
                    }
                )
                
        # 记录总训练时间
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        # 保存训练历史
        with open(self.exp_dir / 'history.json', 'w') as f:
            json.dump(self.history, f)