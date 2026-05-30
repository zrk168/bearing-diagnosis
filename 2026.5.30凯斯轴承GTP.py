# -*- coding: utf-8 -*-
"""
CWRU 轴承故障诊断 | CNN-Transformer 双任务分类
特性: 防震荡训练 / 详细日志 / 早停机制 / 自动可视化 / 单工况适配
"""
import os
import re
import time
import numpy as np
import scipy.io
import matplotlib

matplotlib.use('Agg')  # 非交互环境安全绘图
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════
# 0. 基础配置 & 中文字体
# ══════════════════════════════════════════════════════════
def setup_chinese_font():
    for p in ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']:
        if os.path.exists(p):
            plt.rcParams['font.family'] = fm.FontProperties(fname=p).get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return True
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return False


HAS_CN = setup_chinese_font()

CFG = {
    'data_dir': r'E:\柱塞泵\CWRU轴承数据\cwru_data\0HP',  # 当前仅跑0HP
    'save_dir': r'./cwru_results',
    'seg_len': 1024,
    'overlap': 0.5,
    'batch_size': 64,
    'epochs': 100,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 15,  # 早停耐心值
    'seed': 42,
    'label_smooth': 0.1,  # 防震荡关键
}
os.makedirs(CFG['save_dir'], exist_ok=True)
torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[设备] {DEVICE} | [数据路径] {CFG["data_dir"]}\n')

# ══════════════════════════════════════════════════════════
# 1. 数据加载 (适配CWRU标准命名)
# ══════════════════════════════════════════════════════════
# CWRU 0HP 标准文件名映射 (若你的文件名不同，修改此字典即可)
CWRU_MAP = {
    '97': 'Normal', '98': 'Normal', '99': 'Normal', '100': 'Normal',
    '105': 'Ball_007', '106': 'Ball_007', '107': 'Ball_007', '108': 'Ball_007',
    '109': 'Ball_014', '110': 'Ball_014', '111': 'Ball_014', '112': 'Ball_014',
    '113': 'Ball_021', '114': 'Ball_021', '115': 'Ball_021', '116': 'Ball_021',
    '117': 'IR_007', '118': 'IR_007', '119': 'IR_007', '120': 'IR_007',
    '121': 'IR_014', '122': 'IR_014', '123': 'IR_014', '124': 'IR_014',
    '125': 'IR_021', '126': 'IR_021', '127': 'IR_021', '128': 'IR_021',
    '129': 'OR_007', '130': 'OR_007', '131': 'OR_007', '132': 'OR_007',
    '133': 'OR_014', '134': 'OR_014', '135': 'OR_014', '136': 'OR_014',
    '137': 'OR_021', '138': 'OR_021', '139': 'OR_021', '140': 'OR_021',
}


def parse_label(fname):
    base = os.path.splitext(fname)[0]
    # 优先查表
    if base in CWRU_MAP:
        tag = CWRU_MAP[base]
    else:
        # 正则 fallback
        m = re.search(r'(?i)(normal|ball|inner|outer|ir|or|b)[_\-]?(\d{3})?', base)
        tag = m.group(0) if m else 'Unknown'

    if 'Normal' in tag:
        return 0, 0  # Type:Normal, Sev:Normal
    if 'Ball' in tag:
        return 1, int(tag.split('_')[1]) // 7
    if 'IR' in tag or 'Inner' in tag:
        return 2, int(tag.split('_')[1]) // 7
    if 'OR' in tag or 'Outer' in tag:
        return 3, int(tag.split('_')[1]) // 7
    return -1, -1


def load_signal(fpath):
    mat = scipy.io.loadmat(fpath)
    # 自动寻找包含 DE_time 的变量
    for k in mat:
        if 'DE_time' in k and k[0] != '_':
            return mat[k].flatten().astype(np.float32)
    # 降级：取第一个数值数组
    for k in mat:
        if isinstance(mat[k], np.ndarray) and mat[k].ndim == 2:
            return mat[k].flatten().astype(np.float32)
    raise ValueError(f"无法读取信号: {fpath}")


def slide_window(sig, L, ov):
    step = int(L * (1 - ov))
    return np.array([sig[i:i + L] for i in range(0, len(sig) - L + 1, step)], dtype=np.float32)


def normalize_segments(segs):
    mu = segs.mean(axis=1, keepdims=True)
    std = segs.std(axis=1, keepdims=True) + 1e-8
    return (segs - mu) / std


def build_dataset(data_dir, cfg):
    X, Y_type, Y_sev = [], [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.mat'): continue
        t, s = parse_label(fname)
        if t == -1: continue
        sig = load_signal(os.path.join(data_dir, fname))
        segs = normalize_segments(slide_window(sig, cfg['seg_len'], cfg['overlap']))
        X.append(segs)
        Y_type.extend([t] * len(segs))
        Y_sev.extend([s] * len(segs))
    return np.array(X, dtype=np.float32), np.array(Y_type, dtype=np.int64), np.array(Y_sev, dtype=np.int64)


# ══════════════════════════════════════════════════════════
# 2. Dataset & DataLoader
# ══════════════════════════════════════════════════════════
class BearingDS(Dataset):
    def __init__(self, X, yt, ys):
        self.X = torch.from_numpy(X).unsqueeze(1)  # (N, 1, L)
        self.yt = torch.from_numpy(yt)
        self.ys = torch.from_numpy(ys)

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.yt[i], self.ys[i]


# ══════════════════════════════════════════════════════════
# 3. CNN-Transformer 模型
# ══════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    def __init__(self, ic, oc, k=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ic, oc, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(oc), nn.GELU(),
            nn.Conv1d(oc, oc, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(oc), nn.GELU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x): return self.net(x)


class CNNTransformer(nn.Module):
    def __init__(self, seg_len, n_type=4, n_sev=4):
        super().__init__()
        # 1D CNN 特征提取
        self.cnn = nn.Sequential(
            ConvBlock(1, 32, 7),
            ConvBlock(32, 64, 5),
            ConvBlock(64, 128, 3)
        )
        # 计算Transformer输入序列长度
        with torch.no_grad():
            tmp = self.cnn(torch.zeros(1, 1, seg_len))
            seq_len, d_feat = tmp.shape[2], tmp.shape[1]

        d_model = 64
        self.proj = nn.Linear(d_feat, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
            dropout=0.2, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # 双分类头
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head_type = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.2), nn.Linear(d_model, n_type))
        self.head_sev = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.2), nn.Linear(d_model, n_sev))

    def forward(self, x):
        f = self.cnn(x).permute(0, 2, 1)  # (B, T, C)
        f = self.proj(f) + self.pos_enc  # (B, T, d)
        f = self.transformer(f).permute(0, 2, 1)  # (B, d, T)
        f = self.gap(f).squeeze(-1)  # (B, d)
        return self.head_type(f), self.head_sev(f)


# ══════════════════════════════════════════════════════════
# 4. 训练组件 (防震荡设计)
# ══════════════════════════════════════════════════════════
class SmoothCE(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        n = logits.size(-1)
        log_p = nn.functional.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_p, self.eps / (n - 1))
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.eps)
        return -(smooth * log_p).sum(-1).mean()


class EarlyStopping:
    def __init__(self, patience, min_delta, save_path):
        self.patience, self.min_delta, self.path = patience, min_delta, save_path
        self.best, self.cnt, self.best_ep = np.inf, 0, 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best - self.min_delta:
            self.best, self.cnt, self.best_ep = val_loss, 0, epoch
            torch.save(model.state_dict(), self.path)
            return False
        self.cnt += 1
        return self.cnt >= self.patience


def run_epoch(model, loader, crit, opt, device, train=True):
    model.train() if train else model.eval()
    tot_loss = ct = cs = n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, yt, ys in loader:
            X, yt, ys = X.to(device), yt.to(device), ys.to(device)
            pt, ps = model(X)
            loss = crit(pt, yt) + crit(ps, ys)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防梯度爆炸/震荡
                opt.step()
            tot_loss += loss.item() * len(X)
            ct += (pt.argmax(1) == yt).sum().item()
            cs += (ps.argmax(1) == ys).sum().item()
            n += len(X)
    return tot_loss / n, ct / n, cs / n


# ══════════════════════════════════════════════════════════
# 5. 可视化
# ══════════════════════════════════════════════════════════
def plot_curves(hist, save_dir):
    ep = range(1, len(hist['tl']) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(ep, hist['tl'], label='Train');
    ax[0].plot(ep, hist['vl'], label='Val')
    ax[0].set_title('损失曲线');
    ax[0].set_xlabel('Epoch');
    ax[0].set_ylabel('Loss');
    ax[0].legend();
    ax[0].grid(alpha=0.3)
    ax[1].plot(ep, hist['tat'], label='Type');
    ax[1].plot(ep, hist['tas'], label='Severity')
    ax[1].set_title('训练集准确率');
    ax[1].set_xlabel('Epoch');
    ax[1].set_ylabel('Acc');
    ax[1].legend();
    ax[1].grid(alpha=0.3)
    ax[2].plot(ep, hist['vat'], label='Type');
    ax[2].plot(ep, hist['vas'], label='Severity')
    ax[2].set_title('验证集准确率');
    ax[2].set_xlabel('Epoch');
    ax[2].set_ylabel('Acc');
    ax[2].legend();
    ax[2].grid(alpha=0.3)
    plt.tight_layout();
    plt.savefig(os.path.join(save_dir, 'curves.png'), dpi=150);
    plt.close()


def plot_cm(y_true, y_pred, names, title, fpath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title(title);
    plt.colorbar()
    plt.xticks(range(len(names)), names, rotation=30, ha='right')
    plt.yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout();
    plt.savefig(fpath, dpi=150);
    plt.close()


# ══════════════════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════════════════
def main():
    print('[1/5] 加载数据...')
    X, yt, ys = build_dataset(CFG['data_dir'], CFG)
    print(f'  样本总数: {len(X)} | 类型分布: {dict(zip(*np.unique(yt, return_counts=True)))}')

    # 划分训练/验证集 (分层采样保证类别均衡)
    X_tr, X_vl, yt_tr, yt_vl, ys_tr, ys_vl = train_test_split(
        X, yt, ys, test_size=0.2, random_state=CFG['seed'], stratify=yt)
    print(f'  训练集: {len(X_tr)} | 验证集: {len(X_vl)}')

    tr_loader = DataLoader(BearingDS(X_tr, yt_tr, ys_tr), batch_size=CFG['batch_size'], shuffle=True, num_workers=0)
    vl_loader = DataLoader(BearingDS(X_vl, yt_vl, ys_vl), batch_size=CFG['batch_size'], shuffle=False, num_workers=0)

    type_names = ['Normal', 'Ball', 'InnerRace', 'OuterRace']
    sev_names
