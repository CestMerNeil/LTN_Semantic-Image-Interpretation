import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Dict, List
import cv2

class BaseDataset(Dataset):
    """分割任务的数据集基类"""
    def __init__(
        self,
        root_path: str,
        split: str = 'train',
        img_size: int = 640
    ):
        super().__init__()
        self.root_path = Path(root_path)
        self.split = split
        self.img_size = img_size
        
        # 加载数据配置
        self._load_data_config()
        # 获取图像和标签路径
        self._get_file_paths()
        
    def _load_data_config(self):
        """加载数据配置文件"""
        yaml_file = self.root_path / 'data.yaml'
        if yaml_file.exists():
            with open(yaml_file, 'r') as f:
                self.data_info = yaml.safe_load(f)
                self.num_classes = len(self.data_info['names'])
        else:
            raise FileNotFoundError(f"No data.yaml found in {self.root_path}")
            
    def _get_file_paths(self):
        """获取图像和标签文件路径"""
        raise NotImplementedError("This method should be implemented by child classes")
        
    def _load_image(self, index: int) -> torch.Tensor:
        """加载并预处理图像"""
        img_file = self.img_files[index]
        # 使用cv2读取以保持与标签一致的图像尺寸
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 记录原始尺寸
        self.ori_shape = img.shape
        
        # 调整大小
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 转换为tensor并归一化
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)  # HWC to CHW
        
        return img
        
    def _load_label(self, index: int) -> Dict[str, torch.Tensor]:
        """
        加载分割标签
        返回字典包含:
        - masks: [num_objects, H, W] 二值掩码
        - boxes: [num_objects, 4] 边界框 (x1, y1, x2, y2)
        - classes: [num_objects] 类别索引
        """
        label_file = self.label_files[index]
        
        try:
            # 读取标签文件
            with open(label_file, 'r') as f:
                labels = [x.split() for x in f.read().strip().splitlines()]
            
            if len(labels) == 0:
                return self._create_empty_label()
            
            # 解析标签
            classes = []
            masks = []
            boxes = []
            
            for label in labels:
                # 第一个值是类别
                cls = int(label[0])
                classes.append(cls)
                
                # 解析分割点坐标
                coords = list(map(float, label[1:]))
                pts = np.array(coords).reshape(-1, 2)  # Nx2
                
                # 调整点坐标到目标大小
                pts[:, 0] *= self.img_size
                pts[:, 1] *= self.img_size
                pts = pts.astype(np.int32)
                
                # 创建掩码
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 1)
                masks.append(mask)
                
                # 计算边界框
                x1, y1 = pts.min(0)
                x2, y2 = pts.max(0)
                boxes.append([x1, y1, x2, y2])
            
            # 转换为tensor
            masks = torch.from_numpy(np.stack(masks))
            boxes = torch.tensor(boxes, dtype=torch.float32)
            classes = torch.tensor(classes, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading label {label_file}: {e}")
            return self._create_empty_label()
            
        return {
            'masks': masks,
            'boxes': boxes,
            'classes': classes
        }
    
    def _create_empty_label(self) -> Dict[str, torch.Tensor]:
        """创建空标签"""
        return {
            'masks': torch.zeros((0, self.img_size, self.img_size)),
            'boxes': torch.zeros((0, 4)),
            'classes': torch.zeros(0, dtype=torch.long)
        }
    
    def __len__(self) -> int:
        return len(self.img_files)
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """获取一个数据样本"""
        img = self._load_image(index)
        label = self._load_label(index)
        return img, label

class COCO8Dataset(BaseDataset):
    """COCO8分割数据集"""
    def __init__(
        self,
        root_path: str = "datasets/coco8",
        split: str = 'train',
        img_size: int = 640
    ):
        super().__init__(root_path, split, img_size)
        
    def _get_file_paths(self):
        """设置COCO8特定的文件路径"""
        img_path = self.root_path / 'images' / self.split
        label_path = self.root_path / 'labels' / self.split
        
        self.img_files = sorted(img_path.glob('*.[jJ][pP][gG]'))
        self.img_files.extend(sorted(img_path.glob('*.[pP][nN][gG]')))
        
        self.label_files = [label_path / f"{img_file.stem}.txt" 
                           for img_file in self.img_files]
        
        # 验证文件是否存在
        valid_files = [(img, lbl) for img, lbl in zip(self.img_files, self.label_files)
                      if img.exists() and lbl.exists()]
        if not valid_files:
            raise RuntimeError(f"No valid image-label pairs found in {img_path}")
            
        self.img_files, self.label_files = zip(*valid_files)
        print(f"Loaded {len(self.img_files)} {self.split} images from COCO8")

class CarpartsDataset(BaseDataset):
    """Carports分割数据集"""
    def __init__(
        self,
        root_path: str = "datasets/carparts-seg",
        split: str = 'train',
        img_size: int = 640
    ):
        super().__init__(root_path, split, img_size)
        
    def _get_file_paths(self):
        """设置Carparts特定的文件路径"""
        img_path = self.root_path / self.split / 'images'
        label_path = self.root_path / self.split / 'labels'
        
        self.img_files = sorted(img_path.glob('*.[jJ][pP][gG]'))
        self.img_files.extend(sorted(img_path.glob('*.[pP][nN][gG]')))
        
        self.label_files = [label_path / f"{img_file.stem}.txt" 
                           for img_file in self.img_files]
        
        # 验证文件是否存在
        valid_files = [(img, lbl) for img, lbl in zip(self.img_files, self.label_files)
                      if img.exists() and lbl.exists()]
        if not valid_files:
            raise RuntimeError(f"No valid image-label pairs found in {img_path}")
            
        self.img_files, self.label_files = zip(*valid_files)
        print(f"Loaded {len(self.img_files)} {self.split} images from Carports")


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    自定义的collate函数，处理变长的标签数据
    
    Args:
        batch: List of tuples (image, label_dict)
        
    Returns:
        images: 堆叠的图像张量 [B, C, H, W]
        labels: 标签字典的列表 [B]
    """
    images = []
    labels = []
    
    for img, label in batch:
        images.append(img)
        labels.append(label)
    
    # 堆叠图像
    images = torch.stack(images, 0)
    
    return images, labels


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn
    )