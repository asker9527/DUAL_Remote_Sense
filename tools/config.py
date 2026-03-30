import os

def local_dataset(dataset_name, data_root='./data'):
    """
    根据数据集名称获取训练和测试路径。
    建议将数据集统一存放在项目根目录的 data 文件夹下。
    """
    # 论文中提到的核心数据集: DOTA, DIOR, FGSC-23 [cite: 9, 188]
    dataset_map = {
        'HRSC': ('HRSC/train', 'HRSC/test'),
        'DIOR': ('DIOR/train', 'DIOR/test'),
        'DOTA': ('DOTA/train', 'DOTA/test'),
        'FGSC': ('FGSC/train', 'FGSC/test'),
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(dataset_map.keys())}")

    train_sub, test_sub = dataset_map[dataset_name]
    
    # 使用 os.path.join 保证 Windows/Linux 兼容性
    train_path = os.path.join(data_root, train_sub)
    test_path = os.path.join(data_root, test_sub)
    
    return train_path, test_path

config_sgd = {
    'lr': 0.01,  # 基于批量大小的调整
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'nesterov': False,
    'lr_schedule': 'cosine'
}
