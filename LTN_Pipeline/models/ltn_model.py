import torch
import ltn
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple

class IsObjectPredicate(nn.Module):
    def __init__(self, input_dim: int = 32):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.score_net(x)

class SameClassPredicate(nn.Module):
    def __init__(self, input_dim: int = 32):
        super().__init__()
        self.similarity_net = nn.Sequential(
            nn.Linear(input_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, y):
        return self.similarity_net(torch.cat([x, y], dim=-1))

class LTNNetwork(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 4,
        embedding_dim: int = 32
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim + 2, embedding_dim),  # boxes(4) + class(1) + score(1)
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 定义谓词
        self.isObject = ltn.Predicate(IsObjectPredicate(embedding_dim))
        self.sameClass = ltn.Predicate(SameClassPredicate(embedding_dim))

        # 定义量词和连接词
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregMin(), quantifier="f")
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregMean(), quantifier="e")

        self.And = ltn.Connective(ltn.fuzzy_ops.AndMin())
        self.Or = ltn.Connective(ltn.fuzzy_ops.OrMax())
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())

    def _encode_features(self, features: Dict[str, torch.Tensor], num_obj: int) -> torch.Tensor:
        """将所有特征编码成单个张量"""
        boxes = features['boxes'][:num_obj]  # [N, 4]
        classes = features['classes'][:num_obj].float().unsqueeze(-1)  # [N, 1]
        scores = features['scores'][:num_obj].unsqueeze(-1)  # [N, 1]
        
        # 合并所有特征
        combined = torch.cat([boxes, classes, scores], dim=-1)  # [N, 6]
        
        # 编码特征
        encoded = self.feature_encoder(combined)  # [N, embedding_dim]
        return encoded
    
    def _create_variables(self, features: Dict[str, torch.Tensor]) -> List[ltn.Variable]:
        """从YOLO特征创建LTN变量"""
        batch_vars = []
        
        for b in range(len(features['num_objects'])):
            num_obj = features['num_objects'][b].item()
            
            if num_obj > 0:
                # 编码特征
                encoded_features = self._encode_features({
                    'boxes': features['boxes'][b],
                    'classes': features['classes'][b],
                    'scores': features['scores'][b]
                }, num_obj)
                
                # 创建变量
                vars = ltn.Variable("objects", encoded_features)
                batch_vars.append(vars)
            
        return batch_vars

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 创建LTN变量
        variables = self._create_variables(features)
        satisfactions = []
        
        # 对每张图像应用规则
        for objs in variables:
            # 定义变量 x 和 y
            x = ltn.Variable("x", objs.value)
            y = ltn.Variable("y", objs.value)
            
            # 应用 Forall 和 Exists，直接使用 lambda 函数
            inner_formula = self.And(
                self.isObject(y),
                self.sameClass(x, y)
            )

            exists_formula = self.Exists(y, inner_formula)

            outer_formula = self.Or(
                self.Not(self.isObject(x)),
                exists_formula
            )

            sat = self.Forall(x, outer_formula)
            satisfactions.append(sat.value)
        
        # 合并所有图像的满意度
        if satisfactions:
            total_sat = torch.stack(satisfactions).mean()
        else:
            total_sat = torch.tensor(1.0)
            
        return {
            'satisfaction': total_sat
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算逻辑规则的损失"""
        return -torch.log(outputs['satisfaction'])
