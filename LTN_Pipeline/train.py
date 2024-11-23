import torch
from pathlib import Path
import argparse
import yaml
import logging
from datetime import datetime

from models.feature_extractor import YOLOFeatureExtractor
from models.ltn_model import LTNNetwork
from utils.dataset import COCO8Dataset, CarpartsDataset, create_dataloader
from utils.trainer import Trainer

logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(description='Train LTN...')
    parser.add_argument('--dataset', type=str, default='datasets/carparts-seg',
                        help='path to coco8 dataset')
    parser.add_argument('--yolo-model', type=str, default='pretrained/yolov8n-seg.pt',
                        help='path to pretrained YOLO model')
    parser.add_argument('--img-size', type=int, default=640,
                        help='input image size')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--device', type=str, default='mps',
                        help='device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of worker threads')
    parser.add_argument('--save-dir', type=str, default='runs/',
                        help='directory to save results')
    parser.add_argument('--eval-freq', type=int, default=1,
                        help='validation frequency')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='checkpoint save frequency')
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()

    exp_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(args.save_dir) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )
    
    # 确保路径存在
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    yolo_path = Path(args.yolo_model)
    if not yolo_path.exists():
        logger.error(f"YOLO model not found at {yolo_path}")
        raise FileNotFoundError(f"YOLO model not found at {yolo_path}")
    
    # 设置设备
    device = args.device
    logger.info(f"Using device: {device}")
    
    # 创建数据集
    logger.info("\nCreating datasets...")
    # train_dataset = COCO8Dataset(
    #     root_path=dataset_path,
    #     split="train",
    #     img_size=args.img_size
    # )
    train_dataset = CarpartsDataset(
        root_path=dataset_path,
        split="train",
        img_size=args.img_size
    )
    
    # val_dataset = COCO8Dataset(
    #     root_path=dataset_path,
    #     split="val",
    #     img_size=args.img_size
    # )
    val_dataset = CarpartsDataset(
        root_path=dataset_path,
        split="valid",
        img_size=args.img_size
    )
    
    # 创建数据加载器
    logger.info("\nCreating data loaders...")
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    # 创建模型
    logger.info("\nCreating models...")
    yolo_extractor = YOLOFeatureExtractor( model_path=args.yolo_model )
    
    # 获取类别数量
    with open(dataset_path / 'data.yaml', 'r') as f:
        data_info = yaml.safe_load(f)
    num_classes = len(data_info['names'])
    
    ltn_net = LTNNetwork( num_classes=num_classes )
    
    # 创建优化器
    optimizer = torch.optim.Adam(ltn_net.parameters(), lr=0.001)
    
    # 创建实验名称
    exp_name = f"coco8/coco8_ltn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建训练器
    logger.info("\nCreating trainer...")
    trainer = Trainer(
        yolo_extractor=yolo_extractor,
        ltn_model=ltn_net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir,
        exp_name=exp_name
    )
    
    # 开始训练
    logger.info("\nStarting training...")
    trainer.train(
        num_epochs=args.epochs,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq
    )
    
    logger.info("\nTraining complete!")
    logger.info(f"Results saved at {log_dir}")


if __name__ == '__main__':
    main()