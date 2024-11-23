from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import mean_iou_score

class Evaluator:
    def __init__(self, model, test_loader, config):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        ious = []
        
        for batch in tqdm(self.test_loader, desc='Testing'):
            images = batch['images'].to(self.device)
            masks = batch['masks'].cpu().numpy()
            
            predictions = self.model(images)
            pred_masks = torch.argmax(predictions['seg_pred'], dim=1).cpu().numpy()
            
            # 计算IoU
            for pred, target in zip(pred_masks, masks):
                iou = mean_iou_score(target.flatten(), pred.flatten())
                ious.append(iou)
        
        mean_iou = np.mean(ious)
        print(f'Mean IoU: {mean_iou:.4f}')
        return mean_iou