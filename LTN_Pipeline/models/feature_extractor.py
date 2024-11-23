from pathlib import Path
import torch
from ultralytics import YOLO
from typing import Dict, List, Optional, Union, Tuple
from torch.utils.data import Dataset, DataLoader

class YOLOFeatureExtractor:
    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8n-seg.pt",
        conf_threshold: float = 0.25
    ):
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.num_classes = len(self.class_names)
    
    def _process_result(self, result) -> Dict[str, torch.Tensor]:
        """处理单个YOLO检测结果"""
        H, W = result.orig_shape[:2]
        
        if len(result.boxes) > 0:
            boxes = result.boxes.xyxy      # [N, 4] 格式：x1,y1,x2,y2
            scores = result.boxes.conf     # [N]
            classes = result.boxes.cls     # [N]
            
            # 转换为tensor
            boxes = torch.as_tensor(boxes)
            scores = torch.as_tensor(scores)
            classes = torch.as_tensor(classes)
            
            # 归一化边界框坐标
            norm_boxes = boxes.clone()
            norm_boxes[:, [0, 2]] /= W
            norm_boxes[:, [1, 3]] /= H
            
            centers = (norm_boxes[:, :2] + norm_boxes[:, 2:]) / 2
            widths = norm_boxes[:, 2] - norm_boxes[:, 0]
            heights = norm_boxes[:, 3] - norm_boxes[:, 1]
            
            if hasattr(result, 'masks') and result.masks is not None:
                masks = torch.as_tensor(result.masks.data)
            else:
                masks = torch.zeros((len(boxes), H, W))
        else:
            boxes = torch.zeros((0, 4))
            norm_boxes = torch.zeros((0, 4))
            centers = torch.zeros((0, 2))
            widths = torch.zeros((0,))
            heights = torch.zeros((0,))
            scores = torch.zeros((0,))
            classes = torch.zeros((0,))
            masks = torch.zeros((0, H, W))
        
        return {
            'boxes': norm_boxes,
            'centers': centers,
            'widths': widths,
            'heights': heights,
            'scores': scores,
            'classes': classes,
            'masks': masks,
            'num_objects': torch.tensor(len(boxes)),
            'image_size': torch.tensor([H, W])
        }

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """提取YOLO特征"""
        # 运行YOLO模型
        results = self.model(images, conf=self.conf_threshold)
        
        # 处理每个检测结果
        batch_results = [self._process_result(result) for result in results]
        
        # 计算最大对象数
        max_objects = max(res['num_objects'].item() for res in batch_results)
        batch_size = len(batch_results)
        
        # 初始化输出字典
        output = {}
        
        # 处理所有字段
        for key in ['boxes', 'centers', 'widths', 'heights', 'scores', 'classes', 'masks']:
            # 获取第一个结果的形状来确定维度
            example_tensor = batch_results[0][key]
            tensor_shape = list(example_tensor.shape)
            
            # 创建填充后的张量
            if len(tensor_shape) == 1:  # 1D tensor
                padded = torch.zeros(batch_size, max_objects)
            elif len(tensor_shape) == 2:  # 2D tensor
                padded = torch.zeros(batch_size, max_objects, tensor_shape[1])
            elif len(tensor_shape) == 3:  # 3D tensor (masks)
                padded = torch.zeros(batch_size, max_objects, tensor_shape[1], tensor_shape[2])
            
            # 填充每个批次的结果
            for i, res in enumerate(batch_results):
                num_obj = res['num_objects'].item()
                if num_obj > 0:
                    if len(tensor_shape) == 1:
                        padded[i, :num_obj] = res[key]
                    elif len(tensor_shape) == 2:
                        padded[i, :num_obj, :] = res[key]
                    elif len(tensor_shape) == 3:
                        padded[i, :num_obj, :, :] = res[key]
            
            output[key] = padded
        
        # 添加其他信息
        output['num_objects'] = torch.stack([res['num_objects'] for res in batch_results])
        output['image_size'] = torch.stack([res['image_size'] for res in batch_results])
        
        return output

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'conf_threshold': self.conf_threshold
        }

    def to(self, device: str) -> 'YOLOFeatureExtractor':
        """将模型移动到指定设备"""
        self.model.to(device)
        return self