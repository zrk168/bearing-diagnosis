"""
============================================================
轴向柱塞泵故障诊断 CNN-Transformer 模型
============================================================
数据: pump_sim_dataset (由仿真生成器生成)
任务: 多分类故障诊断 (16类)
模型: CNN特征提取 + Transformer序列建模
输入: 7通道时间序列 (长度20000)
输出: 故障类型 (6类) + 故障程度 (4类) 双头输出
============================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from tqdm import tqdm
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局配置
# ============================================================
CFG = {
    # 数据
    'data_dir':        'pump_sim_dataset',
    'fs':              10000,
    'duration':        2.0,
    'n_channels':      7,          # 修正：实际信号列数
    'seq_len':         20000,

    # 输入分段
    'segment_len':     1000,
    'n_segments':      20,

    # 模型
    'cnn_channels':    [7, 64, 128, 256],   # 第一个数与 n_channels 保持一致
    'cnn_kernel':      7,
    'd_model':         256,
    'n_heads':         8,
    'n_transformer':   4,
    'dim_feedforward': 512,
    'dropout':         0.1,

    # 训练
    'batch_size':      32,
    'epochs':          50,
    'lr':              5e-4,
    'weight_decay':    1e-4,
    'warmup_epochs':   5,
    'patience':        10,
    'grad_clip':       1.0,

    # 标签
    'n_fault_class':    6,
    'n_severity_class': 4,
    'fault_names': [
        'normal', 'slipper_wear', 'loose_slipper',
        'valve_plate_wear', 'piston_wear', 'center_spring_failure',
    ],
    'severity_names': ['normal', 'mild', 'moderate', 'severe'],
    'fault_names_cn':    ['正常', '滑靴磨损', '松靴', '配流盘磨损', '柱塞磨损', '中心弹簧失效'],
    'severity_names_cn': ['正常', '轻度', '中度', '重度'],

    # 输出
    'output_dir': 'cnn_transformer_output',
    'device':     'cuda' if torch.cuda.is_available() else 'cpu',
    'seed':       42,
}



# ============================================================
# 1. 数据集类
# ============================================================
class PumpDataset(Dataset):
    def __init__(self, metadata_df, data_dir, scaler=None, fit_scaler=False):
        self.data_dir = data_dir
        self.meta = metadata_df.reset_index(drop=True)

        # ---- 一次性读取所有数据到内存 ----
        print(f'正在预加载 {len(self.meta)} 个样本到内存...')
        all_signals = []

        for idx in range(len(self.meta)):
            fp = self.meta.loc[idx, 'file_path']
            full_path = os.path.join(data_dir, fp)
            df = pd.read_csv(full_path, header=0)
            signal = df[['pressure_outlet_MPa','pressure_return_MPa','flow_outlet_Lmin','flow_return_Lmin','acc_x_g','acc_y_g','acc_z_g']].values.astype(np.float32)
  # [20000, 7]
            all_signals.append(signal)
        all_signals = np.stack(all_signals, axis=0)  # [N, 20000, 7]

        # ---- 标准化 ----
        if fit_scaler:
            N, T, C = all_signals.shape
            flat = all_signals.reshape(-1, C)
            scaler.fit(flat)

        N, T, C = all_signals.shape
        flat = all_signals.reshape(-1, C)
        flat = scaler.transform(flat)
        all_signals = flat.reshape(N, T, C)

        # 转置为 [N, 7, 20000]
        self.signals = torch.tensor(
            all_signals.transpose(0, 2, 1), dtype=torch.float32
        )

        # ---- 标签 ----
        fault_map = {name: i for i, name in enumerate(CFG['fault_names'])}
        severity_map = {name: i for i, name in enumerate(CFG['severity_names'])}

        self.fault_labels = torch.tensor(
            [fault_map[n] for n in self.meta['fault_type']], dtype=torch.long
        )
        self.severity_labels = torch.tensor(
            [severity_map[n] for n in self.meta['severity']], dtype=torch.long
        )

        print(f'预加载完成，数据形状: {self.signals.shape}')

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # 直接从内存取，不再读磁盘
        return self.signals[idx], self.fault_labels[idx], self.severity_labels[idx]



# ============================================================
# 2. 模型定义
# ============================================================

# ----------------------------------------------------------
# 2.1 CNN 特征提取器
# 输入: [B, C, L] = [B, 7, 20000]
# 输出: [B, n_segments, d_model]
# ----------------------------------------------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=7, d_model=256, segment_len=1000):
        super().__init__()
        self.segment_len = segment_len

        self.conv_block = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),           # L/2

            # Layer 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),           # L/4

            # Layer 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(5),           # L/20
        )

        # 输出: [B, d_model, L/20] = [B, 256, 1000]
        # 再把 L/20 维度作为 sequence 维度送入 Transformer
        # 即 n_segments = seq_len / 20 = 20000 / 20 = 1000

    def forward(self, x):
        # x: [B, C, L]
        out = self.conv_block(x)
        # out: [B, d_model, T]  T = L/20
        out = F.adaptive_avg_pool1d(out, 100)
        out = out.permute(0, 2, 1)
        # out: [B, T, d_model]
        return out


# ----------------------------------------------------------
# 2.2 位置编码
# ----------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)   # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ----------------------------------------------------------
# 2.3 Transformer 编码器
# ----------------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=4,
                 dim_feedforward=512, dropout=0.3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True         # Pre-LN 更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # x: [B, T, d_model]
        return self.encoder(x)


# ----------------------------------------------------------
# 2.4 完整 CNN-Transformer 模型
# 双头输出: fault_type + severity
# ----------------------------------------------------------
class CNNTransformerDiagnostic(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        d_model          = cfg['d_model']
        n_fault_class    = cfg['n_fault_class']
        n_severity_class = cfg['n_severity_class']

        # CNN 提取局部特征
        self.cnn = CNNFeatureExtractor(
            in_channels=cfg['n_channels'],
            d_model=d_model,
            segment_len=cfg['segment_len']
        )

        # 位置编码
        self.pos_enc = PositionalEncoding(
            d_model=d_model,
            max_len=2000,
            dropout=cfg['dropout']
        )

        # Transformer 建模全局依赖
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_transformer'],
            dim_feedforward=cfg['dim_feedforward'],
            dropout=cfg['dropout']
        )

        # 全局平均池化 + 最大池化 拼接
        self.pool_norm = nn.LayerNorm(d_model)

        # 故障类型分类头
        self.fault_head = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(128, n_fault_class)
        )

        # 故障程度分类头
        self.severity_head = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(128, n_severity_class)
        )

    def forward(self, x):
        """
        x: [B, 7, 20000]
        """
        x = self.cnn(x)
        # x: [B, T, d_model]

        x = self.pos_enc(x)
        x = self.transformer(x)
        # x: [B, T, d_model]

        x = self.pool_norm(x)

        # 平均池化 + 最大池化
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)

        feat = torch.cat([avg_pool, max_pool], dim=1)
        # feat: [B, d_model * 2]

        fault_logits = self.fault_head(feat)
        severity_logits = self.severity_head(feat)

        return fault_logits, severity_logits


# ============================================================
# 3. 损失函数
# ============================================================
class DiagnosticLoss(nn.Module):
    def __init__(self, fault_weight=1.0, severity_weight=1.0):
        super().__init__()
        self.ce_fault = nn.CrossEntropyLoss()
        self.ce_severity = nn.CrossEntropyLoss()
        self.fault_weight = fault_weight
        self.severity_weight = severity_weight

    def forward(self, fault_logits, severity_logits, fault_label, severity_label):
        loss_fault = self.ce_fault(fault_logits, fault_label)
        loss_severity = self.ce_severity(severity_logits, severity_label)

        loss = self.fault_weight * loss_fault + self.severity_weight * loss_severity

        return loss, loss_fault, loss_severity


# ============================================================
# 4. 训练和验证函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()

    total_loss = 0.0
    total_fault_loss = 0.0
    total_severity_loss = 0.0

    correct_fault = 0
    correct_severity = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} Train')

    for signals, fault_labels, severity_labels in pbar:
        signals = signals.to(device)
        fault_labels = fault_labels.to(device)
        severity_labels = severity_labels.to(device)

        optimizer.zero_grad()

        fault_logits, severity_logits = model(signals)

        loss, loss_fault, loss_severity = criterion(
            fault_logits,
            severity_logits,
            fault_labels,
            severity_labels
        )

        loss.backward()

        if CFG['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])

        optimizer.step()

        batch_size = signals.size(0)

        total_loss += loss.item() * batch_size
        total_fault_loss += loss_fault.item() * batch_size
        total_severity_loss += loss_severity.item() * batch_size

        pred_fault = torch.argmax(fault_logits, dim=1)
        pred_severity = torch.argmax(severity_logits, dim=1)

        correct_fault += (pred_fault == fault_labels).sum().item()
        correct_severity += (pred_severity == severity_labels).sum().item()
        total += batch_size

        pbar.set_postfix({
            'loss': total_loss / total,
            'fault_acc': correct_fault / total,
            'severity_acc': correct_severity / total
        })

    avg_loss = total_loss / total
    avg_fault_loss = total_fault_loss / total
    avg_severity_loss = total_severity_loss / total
    fault_acc = correct_fault / total
    severity_acc = correct_severity / total

    return avg_loss, avg_fault_loss, avg_severity_loss, fault_acc, severity_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, mode='Val'):
    model.eval()

    total_loss = 0.0
    total_fault_loss = 0.0
    total_severity_loss = 0.0

    correct_fault = 0
    correct_severity = 0
    correct_both = 0
    total = 0

    all_fault_true = []
    all_fault_pred = []
    all_severity_true = []
    all_severity_pred = []

    pbar = tqdm(loader, desc=mode)

    for signals, fault_labels, severity_labels in pbar:
        signals = signals.to(device)
        fault_labels = fault_labels.to(device)
        severity_labels = severity_labels.to(device)

        fault_logits, severity_logits = model(signals)

        loss, loss_fault, loss_severity = criterion(
            fault_logits,
            severity_logits,
            fault_labels,
            severity_labels
        )

        batch_size = signals.size(0)

        total_loss += loss.item() * batch_size
        total_fault_loss += loss_fault.item() * batch_size
        total_severity_loss += loss_severity.item() * batch_size

        pred_fault = torch.argmax(fault_logits, dim=1)
        pred_severity = torch.argmax(severity_logits, dim=1)

        correct_fault += (pred_fault == fault_labels).sum().item()
        correct_severity += (pred_severity == severity_labels).sum().item()
        correct_both += ((pred_fault == fault_labels) &
                         (pred_severity == severity_labels)).sum().item()

        total += batch_size

        all_fault_true.extend(fault_labels.cpu().numpy())
        all_fault_pred.extend(pred_fault.cpu().numpy())
        all_severity_true.extend(severity_labels.cpu().numpy())
        all_severity_pred.extend(pred_severity.cpu().numpy())

        pbar.set_postfix({
            'loss': total_loss / total,
            'fault_acc': correct_fault / total,
            'severity_acc': correct_severity / total,
            'both_acc': correct_both / total
        })

    avg_loss = total_loss / total
    avg_fault_loss = total_fault_loss / total
    avg_severity_loss = total_severity_loss / total

    fault_acc = correct_fault / total
    severity_acc = correct_severity / total
    both_acc = correct_both / total

    results = {
        'loss': avg_loss,
        'fault_loss': avg_fault_loss,
        'severity_loss': avg_severity_loss,
        'fault_acc': fault_acc,
        'severity_acc': severity_acc,
        'both_acc': both_acc,
        'fault_true': np.array(all_fault_true),
        'fault_pred': np.array(all_fault_pred),
        'severity_true': np.array(all_severity_true),
        'severity_pred': np.array(all_severity_pred),
    }

    return results


# ============================================================
# 5. 学习率调度器
# ============================================================
def build_scheduler(optimizer, cfg):
    """
    warmup + cosine decay
    """
    def lr_lambda(epoch):
        if epoch < cfg['warmup_epochs']:
            return float(epoch + 1) / float(cfg['warmup_epochs'])
        else:
            progress = float(epoch - cfg['warmup_epochs']) / \
                       float(max(1, cfg['epochs'] - cfg['warmup_epochs']))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


# ============================================================
# 6. 绘图函数
# ============================================================
def plot_training_curves(history, output_dir):
    epochs = np.arange(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_fault_acc'], label='Train Fault Acc')
    plt.plot(epochs, history['val_fault_acc'], label='Val Fault Acc')
    plt.plot(epochs, history['train_severity_acc'], label='Train Severity Acc')
    plt.plot(epochs, history['val_severity_acc'], label='Val Severity Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f'训练曲线已保存: {save_path}')


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f'混淆矩阵已保存: {save_path}')



# ============================================================
# 7. 路径修正函数
# ============================================================
def fix_metadata_paths(data_dir):
    """
    将 metadata.csv 里的 file_path 统一修正为相对于 data_dir 的路径。

    例如：
    pump_sim_dataset/train/xxx.csv

    修正为：
    train/xxx.csv
    """

    meta_path = os.path.join(data_dir, 'metadata.csv')
    meta = pd.read_csv(meta_path)

    data_dir_norm = str(data_dir).replace('\\', '/').rstrip('/')

    fixed_paths = []

    for fp in meta['file_path']:
        fp = str(fp).replace('\\', '/')

        # 如果是 pump_sim_dataset/train/...，去掉 pump_sim_dataset/
        prefix = data_dir_norm + '/'
        if fp.startswith(prefix):
            fp = fp[len(prefix):]

        # 如果是绝对路径，转成相对路径
        if os.path.isabs(fp):
            try:
                fp = os.path.relpath(fp, data_dir)
                fp = fp.replace('\\', '/')
            except:
                pass

        fixed_paths.append(fp)

    meta['file_path'] = fixed_paths

    fixed_meta_path = os.path.join(data_dir, 'metadata_fixed.csv')
    meta.to_csv(fixed_meta_path, index=False, encoding='utf-8-sig')

    print(f'修正后的 metadata 已保存: {fixed_meta_path}')

    return fixed_meta_path



# ============================================================
# 8. 重新定义 load_metadata，优先读取 metadata_fixed.csv
# ============================================================
def load_metadata(data_dir):
    fixed_meta_path = os.path.join(data_dir, 'metadata_fixed.csv')
    if not os.path.exists(fixed_meta_path):
        fix_metadata_paths(data_dir)

    meta = pd.read_csv(fixed_meta_path)

    # 列名兼容
    if 'file_path' not in meta.columns:
        meta['file_path'] = meta['csv_path']
    if 'fault_type' not in meta.columns:
        meta['fault_type'] = meta['folder'].apply(
            lambda x: 'normal' if x == 'normal' else '_'.join(x.split('_')[:-1])
        )
    if 'severity' not in meta.columns:
        meta['severity'] = meta['folder'].apply(
            lambda x: 'normal' if x == 'normal' else x.split('_')[-1]
        )
    if 'class_name' not in meta.columns:
        meta['class_name'] = meta['folder']

    train_meta = meta[meta['split'] == 'train'].reset_index(drop=True)
    val_meta   = meta[meta['split'] == 'val'  ].reset_index(drop=True)
    test_meta  = meta[meta['split'] == 'test' ].reset_index(drop=True)

    print(f'\n训练集: {len(train_meta)} | 验证集: {len(val_meta)} | 测试集: {len(test_meta)}')
    print('\n训练集类别分布：')
    print(train_meta['class_name'].value_counts().sort_index())

    return train_meta, val_meta, test_meta

# ============================================================
# 9. 构建 DataLoader
# ============================================================
def build_dataloaders(data_dir, batch_size):
    """
    构建训练集、验证集、测试集 DataLoader
    """

    # 读取 metadata
    train_meta, val_meta, test_meta = load_metadata(data_dir)

    # 标准化器：只用训练集拟合
    scaler = StandardScaler()

    print('\n============================================================')
    print('正在构建训练集')
    print('============================================================')

    train_dataset = PumpDataset(
        metadata_df=train_meta,
        data_dir=data_dir,
        scaler=scaler,
        fit_scaler=True
    )

    print('\n============================================================')
    print('正在构建验证集')
    print('============================================================')

    val_dataset = PumpDataset(
        metadata_df=val_meta,
        data_dir=data_dir,
        scaler=scaler,
        fit_scaler=False
    )

    print('\n============================================================')
    print('正在构建测试集')
    print('============================================================')

    test_dataset = PumpDataset(
        metadata_df=test_meta,
        data_dir=data_dir,
        scaler=scaler,
        fit_scaler=False
    )

    # Windows 下 num_workers 建议先设为 0，避免多进程读 CSV 报错
    num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if CFG['device'] == 'cuda' else False,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if CFG['device'] == 'cuda' else False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if CFG['device'] == 'cuda' else False,
        drop_last=False
    )

    print('\nDataLoader 构建完成：')
    print(f'训练批次数: {len(train_loader)}')
    print(f'验证批次数: {len(val_loader)}')
    print(f'测试批次数: {len(test_loader)}')
    import joblib
    joblib.dump(scaler, os.path.join(CFG['output_dir'], 'scaler.pkl'))

    return train_loader, val_loader, test_loader



# ============================================================
# 9. 主训练函数
# ============================================================
def train_model():
    print('\n============================================================')
    print('开始构建数据加载器')
    print('============================================================')

    train_loader, val_loader, test_loader = build_dataloaders(
        CFG['data_dir'],
        CFG['batch_size']
    )

    print('\n============================================================')
    print('开始构建模型')
    print('============================================================')

    model = CNNTransformerDiagnostic(CFG).to(CFG['device'])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print(f'总参数量: {total_params:,}')
    print(f'可训练参数量: {trainable_params:,}')

    criterion = DiagnosticLoss(
        fault_weight=1.0,
        severity_weight=1.0
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG['lr'],
        weight_decay=CFG['weight_decay']
    )

    scheduler = build_scheduler(optimizer, CFG)

    history = {
        'train_loss': [],
        'train_fault_loss': [],
        'train_severity_loss': [],
        'train_fault_acc': [],
        'train_severity_acc': [],

        'val_loss': [],
        'val_fault_loss': [],
        'val_severity_loss': [],
        'val_fault_acc': [],
        'val_severity_acc': [],
        'val_both_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    best_model_path = os.path.join(CFG['output_dir'], 'best_cnn_transformer.pth')
    last_model_path = os.path.join(CFG['output_dir'], 'last_cnn_transformer.pth')

    print('\n============================================================')
    print('开始训练')
    print('============================================================')

    for epoch in range(1, CFG['epochs'] + 1):
        current_lr = optimizer.param_groups[0]['lr']

        print(f'\nEpoch [{epoch}/{CFG["epochs"]}]  LR: {current_lr:.6e}')

        train_loss, train_fault_loss, train_severity_loss, train_fault_acc, train_severity_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            CFG['device'],
            epoch
        )

        val_results = evaluate(
            model,
            val_loader,
            criterion,
            CFG['device'],
            mode='Val'
        )

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_fault_loss'].append(train_fault_loss)
        history['train_severity_loss'].append(train_severity_loss)
        history['train_fault_acc'].append(train_fault_acc)
        history['train_severity_acc'].append(train_severity_acc)

        history['val_loss'].append(val_results['loss'])
        history['val_fault_loss'].append(val_results['fault_loss'])
        history['val_severity_loss'].append(val_results['severity_loss'])
        history['val_fault_acc'].append(val_results['fault_acc'])
        history['val_severity_acc'].append(val_results['severity_acc'])
        history['val_both_acc'].append(val_results['both_acc'])
        history['lr'].append(current_lr)

        print(f'Train Loss: {train_loss:.4f} | '
              f'Train Fault Acc: {train_fault_acc:.4f} | '
              f'Train Severity Acc: {train_severity_acc:.4f}')

        print(f'Val Loss: {val_results["loss"]:.4f} | '
              f'Val Fault Acc: {val_results["fault_acc"]:.4f} | '
              f'Val Severity Acc: {val_results["severity_acc"]:.4f} | '
              f'Val Both Acc: {val_results["both_acc"]:.4f}')

        # 这里用 fault_acc 和 severity_acc 的平均值作为保存标准
        val_score = 0.5 * val_results['fault_acc'] + 0.5 * val_results['severity_acc']

        if val_score > best_val_acc:
            best_val_acc = val_score
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'cfg': CFG
            }, best_model_path)

            print(f'保存最优模型: {best_model_path}')
        else:
            patience_counter += 1
            print(f'验证集未提升，patience: {patience_counter}/{CFG["patience"]}')

        if patience_counter >= CFG['patience']:
            print('\n触发早停')
            break

    # 保存最后一个模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'cfg': CFG
    }, last_model_path)

    print(f'\n最后模型已保存: {last_model_path}')

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_path = os.path.join(CFG['output_dir'], 'training_history.csv')
    history_df.to_csv(history_path, index=False, encoding='utf-8-sig')
    print(f'训练历史已保存: {history_path}')

    plot_training_curves(history, CFG['output_dir'])

    print('\n============================================================')
    print('开始测试最优模型')
    print('============================================================')

    checkpoint = torch.load(best_model_path, map_location=CFG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = evaluate(
        model,
        test_loader,
        criterion,
        CFG['device'],
        mode='Test'
    )

    print('\n============================================================')
    print('测试集结果')
    print('============================================================')
    print(f'Best Epoch: {best_epoch}')
    print(f'Test Loss: {test_results["loss"]:.4f}')
    print(f'Test Fault Acc: {test_results["fault_acc"]:.4f}')
    print(f'Test Severity Acc: {test_results["severity_acc"]:.4f}')
    print(f'Test Both Acc: {test_results["both_acc"]:.4f}')

    # 故障类型分类报告
    fault_report = classification_report(
        test_results['fault_true'],
        test_results['fault_pred'],
        target_names=CFG['fault_names'],
        digits=4
    )

    print('\n故障类型分类报告：')
    print(fault_report)

    with open(os.path.join(CFG['output_dir'], 'fault_classification_report.txt'),
              'w', encoding='utf-8') as f:
        f.write(fault_report)

    # 故障程度分类报告
    severity_report = classification_report(
        test_results['severity_true'],
        test_results['severity_pred'],
        target_names=CFG['severity_names'],
        digits=4
    )

    print('\n故障程度分类报告：')
    print(severity_report)

    with open(os.path.join(CFG['output_dir'], 'severity_classification_report.txt'),
              'w', encoding='utf-8') as f:
        f.write(severity_report)

    # 混淆矩阵
    plot_confusion_matrix(
        test_results['fault_true'],
        test_results['fault_pred'],
        CFG['fault_names'],
        'Fault Type Confusion Matrix',
        os.path.join(CFG['output_dir'], 'fault_confusion_matrix.png')
    )

    plot_confusion_matrix(
        test_results['severity_true'],
        test_results['severity_pred'],
        CFG['severity_names'],
        'Severity Confusion Matrix',
        os.path.join(CFG['output_dir'], 'severity_confusion_matrix.png')
    )

    # 保存测试预测结果
    pred_df = pd.DataFrame({
        'fault_true': test_results['fault_true'],
        'fault_pred': test_results['fault_pred'],
        'fault_true_name': [CFG['fault_names'][i] for i in test_results['fault_true']],
        'fault_pred_name': [CFG['fault_names'][i] for i in test_results['fault_pred']],
        'severity_true': test_results['severity_true'],
        'severity_pred': test_results['severity_pred'],
        'severity_true_name': [CFG['severity_names'][i] for i in test_results['severity_true']],
        'severity_pred_name': [CFG['severity_names'][i] for i in test_results['severity_pred']],
    })

    pred_path = os.path.join(CFG['output_dir'], 'test_predictions.csv')
    pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
    print(f'测试集预测结果已保存: {pred_path}')

    print('\n============================================================')
    print('训练和测试完成')
    print('============================================================')

    return model, history, test_results


# ============================================================
# 10. 单样本预测函数
# ============================================================
@torch.no_grad()
def predict_one_csv(model_path, csv_path, device=None):
    """
    对单个 CSV 样本进行预测
    """
    if device is None:
        device = CFG['device']

    model = CNNTransformerDiagnostic(CFG).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    df = pd.read_csv(csv_path, header=0)
    SIGNAL_COLS = ['pressure_outlet_MPa', 'pressure_return_MPa', 'flow_outlet_Lmin', 'flow_return_Lmin', 'acc_x_g',
                   'acc_y_g', 'acc_z_g']
    signal = df[SIGNAL_COLS].values.astype(np.float32)

    # 单样本预测时，这里仅做简单标准化
    # 更严格的做法是保存训练集 scaler 后再加载使用
    sig = (sig - sig.mean(axis=0, keepdims=True)) / (sig.std(axis=0, keepdims=True) + 1e-8)

    sig = sig.T

    L = CFG['seq_len']
    if sig.shape[1] >= L:
        sig = sig[:, :L]
    else:
        pad = np.zeros((sig.shape[0], L - sig.shape[1]), dtype=np.float32)
        sig = np.concatenate([sig, pad], axis=1)

    x = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).to(device)

    fault_logits, severity_logits = model(x)

    fault_prob = torch.softmax(fault_logits, dim=1).cpu().numpy()[0]
    severity_prob = torch.softmax(severity_logits, dim=1).cpu().numpy()[0]

    fault_pred = int(np.argmax(fault_prob))
    severity_pred = int(np.argmax(severity_prob))

    result = {
        'fault_pred': fault_pred,
        'fault_name': CFG['fault_names'][fault_pred],
        'fault_name_cn': CFG['fault_names_cn'][fault_pred],
        'fault_prob': fault_prob[fault_pred],

        'severity_pred': severity_pred,
        'severity_name': CFG['severity_names'][severity_pred],
        'severity_name_cn': CFG['severity_names_cn'][severity_pred],
        'severity_prob': severity_prob[severity_pred],

        'fault_prob_all': fault_prob,
        'severity_prob_all': severity_prob
    }

    return result


# ============================================================
# 11. 主程序入口
# ============================================================
if __name__ == '__main__':

    model, history, test_results = train_model()

    # 如果你想测试单个 CSV，可以取消下面注释
    """
    model_path = os.path.join(CFG['output_dir'], 'best_cnn_transformer.pth')

    sample_csv = 'pump_sim_dataset/test/slipper_wear_mild/10MPa/test_slipper_wear_mild_10MPa_0000.csv'

    result = predict_one_csv(model_path, sample_csv)

    print('单样本预测结果:')
    print(result)
    """
