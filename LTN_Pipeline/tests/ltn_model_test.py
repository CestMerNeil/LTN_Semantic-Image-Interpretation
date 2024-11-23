from models.ltn_model import LTNNetwork
from models.feature_extractor import YOLOFeatureExtractor
import torch

def ltn_model_test():
    # 初始化 LTN 网络和特征提取器
    device = "mps"
    ltn_net = LTNNetwork(
        num_classes=80,
        device=device
    )
    
    extractor = YOLOFeatureExtractor(
        model_path="pretrained/yolov8n-seg.pt",
        conf_threshold=0.25,
        device=device
    )
    
    # 创建测试图像
    # 确保值在[0,1]范围内
    images = torch.rand(4, 3, 640, 640, device=device)
    
    # 提取YOLO特征
    print("\nExtracting YOLO features...")
    yolo_features = extractor.extract_features(images)
    print(f"Number of detected objects per image: {yolo_features['num_objects']}")
    
    # 应用LTN网络
    print("\nApplying LTN rules...")
    outputs = ltn_net(yolo_features)
    
    # 计算损失
    loss = ltn_net.compute_loss(outputs)
    
    print("\nResults:")
    print(f"Loss: {loss.item():.4f}")
    print(f"Rule satisfaction: {outputs['satisfaction'].item():.4f}")
    
    # 打印每个特征的形状
    print("\nFeature shapes:")
    for key, value in yolo_features.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
    
    return {
        "loss": loss.item(),
        "outputs": outputs,
        "yolo_features": yolo_features
    }

if __name__ == "__main__":
    results = ltn_model_test()