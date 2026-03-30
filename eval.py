import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.special import digamma

# 导入自定义工具包 (确保路径正确)
from tools import local_dataset, set_output_layer

def parse_args():
    parser = argparse.ArgumentParser(description='EDL Uncertainty Testing Script')
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--dataset', type=str, default='FGSC', help='Dataset name')
    parser.add_argument('--method', type=str, default='trust_decomposition', help='Method name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory containing .pth files')
    parser.add_argument('--save_dir', type=str, default='./test_results', help='Directory to save CSVs')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index of checkpoints')
    parser.add_argument('--end_idx', type=int, default=100, help='End index of checkpoints')
    return parser.parse_args()

def edl_uncertainty_decomposition(alpha):
    """
    计算基于熵的 EDL 不确定性分解 (支持 Batch)
    alpha: Tensor, shape (B, K), 对应 EDL 的证据参数 (softplus(logits) + 1)
    """
    K = alpha.shape[1]
    S = torch.sum(alpha, dim=1, keepdim=True)  # (B, 1)
    probs = alpha / S  # E[p], shape (B, K)
    
    # 1. Total Entropy (Up)
    # 增加 epsilon 防止 log(0)
    total_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

    # 2. Aleatoric Uncertainty (Ua)
    # 依据公式: E[H(p|theta)]
    psi_S = digamma(S + 1)
    psi_alpha = digamma(alpha + 1)
    aleatoric = torch.sum(probs * (psi_S - psi_alpha), dim=1)

    # 3. Epistemic Uncertainty (Ue)
    epistemic = total_entropy - aleatoric

    # 4. K/S (传统 EDL 衡量指标)
    ks_metric = K / S.squeeze(1)

    return total_entropy, epistemic, aleatoric, ks_metric

def run_inference(model, dataloader, device, num_classes):
    """运行推理并收集结果"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) # 在 EDL 中 outputs 即为 alpha
            
            preds = torch.argmax(outputs, dim=1)
            up, ue, ua, ks = edl_uncertainty_decomposition(outputs)
            
            for i in range(len(labels)):
                results.append({
                    'label': labels[i].item(),
                    'pred': preds[i].item(),
                    'correct': (preds[i] == labels[i]).item(),
                    'Up': up[i].item(),
                    'Ue': ue[i].item(),
                    'Ua': ua[i].item(),
                    'K/s': ks[i].item()
                })
    return results

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 路径准备
    os.makedirs(args.save_dir, exist_ok=True)

    # 2. 数据准备 (只需加载一次)
    image_size, crop_size = 512, 448
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    train_path, test_path = local_dataset(args.dataset)
    train_dataset = datasets.ImageFolder(root=train_path, transform=test_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    num_classes = len(train_dataset.classes)
    print(f"Dataset: {args.dataset} | Classes: {num_classes}")

    # 3. 模型结构定义 (只需定义一次)
    if args.model == 'resnet50':
        model = models.resnet50(weights=None)
    else:
        raise NotImplementedError("Only resnet50 is configured in this script.")