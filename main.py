import os
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 导入自定义工具包
from tools import get_loss, local_dataset, configure_optimizer, CMO_weighted_train_loader
from tools import rand_bbox, set_output_layer

def parse_args():
    parser = argparse.ArgumentParser(description='DUAL Framework for Long-tailed Remote Sensing Classification')
    
    # 基础配置
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'mobilenet_v2', 'efficientnet_b0'], help='Backbone model name')
    parser.add_argument('--dataset', type=str, default='FGSC', choices=['FGSC', 'DIOR', 'DOTA'], help='Dataset name')
    parser.add_argument('--method', type=str, default='trust_decomposition', help='Training method (CE, trust, trust_decomposition, etc.)')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of total epochs to run')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for cosine annealing')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of warmup epochs')
    
    # DUAL/EDL 特定参数
    parser.add_argument('--activation', type=str, default='softplus', help='Activation for evidential layer')
    parser.add_argument('--sigma', type=float, default=3.0, help='Exponential scaling factor for EU reweighting')
    parser.add_argument('--lambda_bal', type=float, default=0.2, help='Balancing factor for KL divergence')
    
    # 路径配置
    parser.add_argument('--output_dir', type=str, default='./output', help='Root directory for outputs')
    
    return parser.parse_args()

def setup_data(args):
    """准备数据增强、数据集和加载器"""
    image_size, crop_size = 512, 448
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(crop_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_path, test_path = local_dataset(args.dataset)
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, train_dataset

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 自动生成实验保存路径
    exp_name = f"{args.dataset}_{args.model}_{args.method}_sig{args.sigma}_pre{args.pretrained}"
    save_path = os.path.join(args.output_dir, exp_name)
    models_dir = os.path.join(save_path, 'models')
    logs_dir = os.path.join(save_path, 'logs')
    results_dir = os.path.join(save_path, 'results')
    for d in [models_dir, logs_dir, results_dir]: os.makedirs(d, exist_ok=True)

    # 2. 数据准备
    train_loader, test_loader, train_dataset = setup_data(args)
    num_classes = len(train_dataset.classes)
    cls_num_list = list(Counter(train_dataset.targets).values())
    
    # CMO 加载器
    weighted_train_loader = None
    if 'cmo' in args.method.lower():
        weighted_train_loader = CMO_weighted_train_loader(
            cls_num_list=cls_num_list, train_dataset=train_dataset, 
            batch_size=args.batch_size, weighted_alpha=1
        )

    # 3. 模型构建
    model_ft = {
        'resnet50': models.resnet50,
        'resnet18': models.resnet18,
        'mobilenet_v2': models.mobilenet_v2,
        'efficientnet_b0': models.efficientnet_b0
    }[args.model](pretrained=args.pretrained)
    
    model = set_output_layer(model_ft, num_classes, args.method, activation=args.activation).to(device)

    # 4. 优化器与损失函数
    criterion = get_loss(args.method)
    warmup_epochs = args.epochs * args.warmup_ratio
    optimizer, scheduler = configure_optimizer(
        model, args.method, args.lr, args.min_lr, args.epochs, warmup_epochs
    )

    writer = SummaryWriter(log_dir=logs_dir)
    best_acc, best_avg_acc = 0.0, 0.0

    # 5. 训练循环
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        
        # 处理 CMO 迭代器
        if weighted_train_loader:
            weighted_iter = iter(weighted_train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            bs = inputs.size(0)

            # --- CMO 数据增强逻辑 ---
            cmo_gate = False
            lam = 1.0
            if weighted_train_loader and warmup_epochs < epoch < args.epochs - warmup_epochs and np.random.rand() < 0.5:
                try:
                    input2, target2 = next(weighted_iter)
                except StopIteration:
                    weighted_iter = iter(weighted_train_loader)
                    input2, target2 = next(weighted_iter)
                
                input2, target2 = input2[:bs].to(device), target2[:bs].to(device)
                lam = np.random.beta(1.0, 1.0)
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                cmo_gate = True

            # --- 前向传播与损失计算 ---
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if 'trust' in args.method:
                # 假设 EDL 损失返回 loss 以及四个分量 A, B, C, D
                loss, *components = criterion(labels, outputs, num_classes, epoch, args.epochs)
            elif cmo_gate:
                loss = criterion(outputs, labels) * lam + criterion(outputs, target2) * (1. - lam)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # --- 每个 Epoch 结束后的逻辑 ---
        if scheduler: scheduler.step()
        
        train_acc = 100. * correct / total
        avg_cls_acc = calculate_avg_cls_acc(all_labels, all_preds)
        
        # 记录训练日志
        writer.add_scalar('Loss/train', running_loss/len(train_loader), epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)

        # --- 评估 ---
        test_acc, test_avg_acc, cls_accs = evaluate(model, test_loader, device)
        writer.add_scalar('Acc/test', test_acc, epoch)
        writer.add_scalar('AvgAcc/test', test_avg_acc, epoch)

        print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%, Test Avg Acc {test_avg_acc:.2f}%")

        # --- 保存最佳模型 ---
        if test_acc >= best_acc:
            best_acc = test_acc
            best_avg_acc = test_avg_acc
            save_results(results_dir, epoch, test_acc, test_avg_acc, cls_accs)
            torch.save(model.state_dict(), os.path.join(models_dir, f'{args.model}_best.pth'))

    writer.close()

def calculate_avg_cls_acc(labels, preds):
    labels, preds = np.array(labels), np.array(preds)
    cls_accs = []
    for cls in np.unique(labels):
        mask = (labels == cls)
        cls_accs.append(100. * np.sum(preds[mask] == labels[mask]) / np.sum(mask))
    return np.mean(cls_accs)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    labels_np, preds_np = np.array(all_labels), np.array(all_preds)
    unique_classes = np.unique(labels_np)
    cls_accs = {}
    for cls in unique_classes:
        mask = (labels_np == cls)
        cls_accs[cls] = 100. * np.sum(preds_np[mask] == labels_np[mask]) / np.sum(mask)
    
    overall_acc = 100. * np.sum(preds_np == labels_np) / len(labels_np)
    avg_cls_acc = np.mean(list(cls_accs.values()))
    return overall_acc, avg_cls_acc, cls_accs

def save_results(path, epoch, acc, avg_acc, cls_accs):
    with open(os.path.join(path, 'results.txt'), 'a') as f:
        f.write(f"Epoch {epoch+1} | Acc: {acc:.2f}% | Avg Acc: {avg_acc:.2f}%\n")
        for cls, val in cls_accs.items():
            f.write(f"  Class {cls}: {val:.2f}%\n")
        f.write("-" * 30 + "\n")

if __name__ == '__main__':
    main()