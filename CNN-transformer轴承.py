# -*- coding: utf-8 -*-
"""
CWRU 轴承故障诊断 —— CNN-Transformer 双任务分类
改进点:
  1. 按时间顺序划分训练/测试集，消除滑窗数据泄露
  2. 跨工况泛化测试 (0HP训练 → 1HP/2HP/3HP测试)
  3. 训练时加入高斯噪声增强
  4. 缩小模型容量，与数据规模匹配
  5. 损失曲线尖峰修复：改用 CosineAnnealingLR
"""

import os, re, time
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# 0. 中文字体
# ══════════════════════════════════════════════════════════
def setup_font():
    for p in ['C:/Windows/Fonts/simhei.ttf',
              'C:/Windows/Fonts/msyh.ttc',
              'C:/Windows/Fonts/simsun.ttc']:
        if os.path.exists(p):
            plt.rcParams['font.family'] = fm.FontProperties(fname=p).get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return True
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return False

HAS_CN = setup_font()

# ══════════════════════════════════════════════════════════
# 1. 超参数
# ══════════════════════════════════════════════════════════
CFG = {
    # ── 路径 ──────────────────────────────────────────────
    'train_hp'    : [r'E:\柱塞泵\CWRU轴承数据\cwru_data\0HP'],   # 训练工况
    'test_hp'     : [r'E:\柱塞泵\CWRU轴承数据\cwru_data\1HP',    # 跨工况测试
                     r'E:\柱塞泵\CWRU轴承数据\cwru_data\2HP',
                     r'E:\柱塞泵\CWRU轴承数据\cwru_data\3HP'],
    'save_dir'    : r'E:\柱塞泵\CWRU轴承数据\results_v2',
    # ── 信号处理 ──────────────────────────────────────────
    'seg_len'     : 1024,
    'overlap'     : 0.5,
    # 时间顺序划分比例（在单个文件内按时间切割，不随机打乱）
    'train_ratio' : 0.70,   # 前70%时间段 → 训练
    'val_ratio'   : 0.15,   # 中15%时间段 → 验证
    # 剩余15%                → 同工况测试
    # ── 训练 ──────────────────────────────────────────────
    'batch_size'  : 64,
    'epochs'      : 100,
    'lr'          : 1e-3,
    'weight_decay': 1e-4,
    'patience'    : 15,
    'min_delta'   : 1e-4,
    'seed'        : 42,
    # ── 模型（缩小容量）──────────────────────────────────
    'cnn_ch'      : [1, 16, 32, 64],   # 原来 [1,32,64,128]
    'd_model'     : 64,                 # 原来 128
    'nhead'       : 4,
    'tf_layers'   : 1,                  # 原来 2
    'dropout'     : 0.3,                # 原来 0.2，略微增大
    'label_smooth': 0.1,
    # ── 噪声增强 ──────────────────────────────────────────
    'noise_std'   : 0.05,               # 训练时叠加高斯噪声标准差
}

os.makedirs(CFG['save_dir'], exist_ok=True)
torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════════
# 2. 数据加载（按时间顺序划分，无泄露）
# ══════════════════════════════════════════════════════════
SEV_NAME = {'007': '轻度', '014': '中度', '021': '重度'}
SEV_CODE = {'007': 1, '014': 2, '021': 3}

def parse_fname(fname):
    base = os.path.splitext(fname)[0].lower()
    if base.startswith('normal'):
        return 'Normal', None
    m = re.search(r'_(b|ir|or)(\d{3})', base)
    if m:
        k = m.group(1).upper()
        s = m.group(2)
        t = {'B': 'Ball', 'IR': 'InnerRace', 'OR': 'OuterRace'}[k]
        return t, s
    return None, None

def read_de(fpath):
    mat = scipy.io.loadmat(fpath)
    key = next(k for k in mat if 'DE_time' in k)
    return mat[key].flatten().astype(np.float32)

def slide(sig, L, ov):
    step = int(L * (1 - ov))
    return np.array([sig[i:i+L] for i in range(0, len(sig)-L+1, step)],
                    dtype=np.float32)

def zscore(segs):
    mu  = segs.mean(1, keepdims=True)
    std = segs.std(1,  keepdims=True) + 1e-8
    return (segs - mu) / std

def load_hp_dir(data_dir, cfg, split='all'):
    """
    从一个工况目录加载数据。
    split:
      'train' → 每个文件前 train_ratio 的时间段
      'val'   → 中间 val_ratio 的时间段
      'same_test' → 最后 (1-train_ratio-val_ratio) 的时间段
      'all'   → 全部（用于跨工况测试目录）
    返回: X (N,1,L), yt (N,), ys (N,)，以及 type_str list
    """
    L   = cfg['seg_len']
    ov  = cfg['overlap']
    tr  = cfg['train_ratio']
    vr  = cfg['val_ratio']

    Xs, Yt, Ys = [], [], []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.mat'):
            continue
        ftype, sev = parse_fname(fname)
        if ftype is None:
            continue

        sig = read_de(os.path.join(data_dir, fname))
        n   = len(sig)

        # ── 按时间顺序切割信号 ──────────────────────────
        if split == 'train':
            seg_sig = sig[:int(n * tr)]
        elif split == 'val':
            seg_sig = sig[int(n * tr): int(n * (tr + vr))]
        elif split == 'same_test':
            seg_sig = sig[int(n * (tr + vr)):]
        else:  # 'all'
            seg_sig = sig

        if len(seg_sig) < L:
            continue

        segs = zscore(slide(seg_sig, L, ov))
        sc   = 0 if sev is None else SEV_CODE[sev]

        Xs.append(segs)
        Yt.extend([ftype] * len(segs))
        Ys.extend([sc]    * len(segs))

    if not Xs:
        return None, None, None

    X  = np.concatenate(Xs)[:, np.newaxis, :]
    return X, np.array(Yt), np.array(Ys, dtype=np.int64)


def build_datasets(cfg):
    """
    构建三份数据集：
      train_set   : 训练工况 前70%时间
      val_set     : 训练工况 中15%时间
      same_test   : 训练工况 后15%时间  （同工况测试）
      cross_test  : 其他工况 全部        （跨工况测试）
    同时拟合 LabelEncoder，保证所有集合标签一致。
    """
    print('\n' + '='*70)
    print('  数据集构建（按时间顺序划分，消除滑窗泄露）')
    print('='*70)

    # ── 训练工况 ──────────────────────────────────────────
    Xtr, Ytr_s, Ytr_sv = [], [], []
    Xvl, Yvl_s, Yvl_sv = [], [], []
    Xst, Yst_s, Yst_sv = [], [], []

    for hp_dir in cfg['train_hp']:
        if not os.path.exists(hp_dir):
            print(f'  [跳过] 目录不存在: {hp_dir}')
            continue
        print(f'\n  [训练工况] {hp_dir}')
        for split, Xl, Yl, Ysl in [
            ('train',     Xtr, Ytr_s, Ytr_sv),
            ('val',       Xvl, Yvl_s, Yvl_sv),
            ('same_test', Xst, Yst_s, Yst_sv),
        ]:
            X, Yt, Ys = load_hp_dir(hp_dir, cfg, split=split)
            if X is not None:
                Xl.append(X); Yl.extend(Yt); Ysl.extend(Ys)

    # ── 跨工况测试 ────────────────────────────────────────
    Xct, Yct_s, Yct_sv = [], [], []
    for hp_dir in cfg['test_hp']:
        if not os.path.exists(hp_dir):
            print(f'  [跳过] 目录不存在: {hp_dir}')
            continue
        print(f'  [跨工况测试] {hp_dir}')
        X, Yt, Ys = load_hp_dir(hp_dir, cfg, split='all')
        if X is not None:
            Xct.append(X); Yct_s.extend(Yt); Yct_sv.extend(Ys)

    # ── 拟合 LabelEncoder（仅用训练集标签）──────────────
    le = LabelEncoder()
    le.fit(Ytr_s)
    type_names = list(le.classes_)
    sev_names  = ['Normal', '轻度(007)', '中度(014)', '重度(021)']

    def encode(Xl, Yl, Ysl):
        if not Xl:
            return None, None, None
        X  = np.concatenate(Xl)
        yt = le.transform(np.array(Yl)).astype(np.int64)
        ys = np.array(Ysl, dtype=np.int64)
        return X, yt, ys

    X_tr, yt_tr, ys_tr = encode(Xtr, Ytr_s, Ytr_sv)
    X_vl, yt_vl, ys_vl = encode(Xvl, Yvl_s, Yvl_sv)
    X_st, yt_st, ys_st = encode(Xst, Yst_s, Yst_sv)
    X_ct, yt_ct, ys_ct = encode(Xct, Yct_s, Yct_sv)

    # ── 打印统计 ──────────────────────────────────────────
    print('\n' + '-'*70)
    for name, X, yt in [('训练集  ', X_tr, yt_tr),
                         ('验证集  ', X_vl, yt_vl),
                         ('同工况测试', X_st, yt_st),
                         ('跨工况测试', X_ct, yt_ct)]:
        if X is None:
            print(f'  {name}: 无数据')
            continue
        dist = {type_names[i]: int((yt==i).sum()) for i in range(len(type_names))}
        print(f'  {name}: {len(X):5d} 样本  分布={dist}')
    print('='*70)

    return (X_tr, yt_tr, ys_tr,
            X_vl, yt_vl, ys_vl,
            X_st, yt_st, ys_st,
            X_ct, yt_ct, ys_ct,
            type_names, sev_names)

# ══════════════════════════════════════════════════════════
# 3. Dataset（训练集支持噪声增强）
# ══════════════════════════════════════════════════════════
class BearingDS(Dataset):
    def __init__(self, X, yt, ys, noise_std=0.0):
        self.X         = torch.from_numpy(X)
        self.yt        = torch.from_numpy(yt)
        self.ys        = torch.from_numpy(ys)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, self.yt[i], self.ys[i]

# ══════════════════════════════════════════════════════════
# 4. 模型（缩小容量）
# ══════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    def __init__(self, ic, oc, k=7):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(ic, oc, k, padding=p, bias=False),
            nn.BatchNorm1d(oc), nn.GELU(),
            nn.Conv1d(oc, oc, k, padding=p, bias=False),
            nn.BatchNorm1d(oc), nn.GELU(),
            nn.MaxPool1d(2),
        )
    def forward(self, x):
        return self.net(x)

class PosEnc(nn.Module):
    def __init__(self, d, maxlen=512, drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe  = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])

class CNNTransformer(nn.Module):
    def __init__(self, seg_len, cnn_ch, d_model, nhead,
                 tf_layers, dropout, n_type, n_sev):
        super().__init__()
        self.cnn = nn.Sequential(
            *[ConvBlock(cnn_ch[i], cnn_ch[i+1]) for i in range(len(cnn_ch)-1)]
        )
        with torch.no_grad():
            tmp  = self.cnn(torch.zeros(1, 1, seg_len))
            c    = tmp.shape[1]
            tlen = tmp.shape[2]

        self.proj = nn.Linear(c, d_model)
        self.pe   = PosEnc(d_model, maxlen=tlen + 16, drop=dropout)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.tf  = nn.TransformerEncoder(enc, num_layers=tf_layers)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.head_type = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_type))
        self.head_sev  = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_sev))

    def forward(self, x):
        f = self.cnn(x).permute(0, 2, 1)    # (B, T', C)
        f = self.pe(self.proj(f))            # (B, T', d)
        f = self.tf(f).permute(0, 2, 1)     # (B, d, T')
        f = self.gap(f).squeeze(-1)          # (B, d)
        return self.head_type(f), self.head_sev(f)

# ══════════════════════════════════════════════════════════
# 5. 标签平滑损失
# ══════════════════════════════════════════════════════════
class SmoothCE(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        n  = logits.size(-1)
        lp = nn.functional.log_softmax(logits, dim=-1)
        with torch.no_grad():
            st = torch.full_like(lp, self.eps / (n - 1))
            st.scatter_(1, target.unsqueeze(1), 1.0 - self.eps)
        return -(st * lp).sum(-1).mean()

# ══════════════════════════════════════════════════════════
# 6. 早停
# ══════════════════════════════════════════════════════════
class EarlyStopping:
    def __init__(self, patience, min_delta, path):
        self.patience  = patience
        self.min_delta = min_delta
        self.path      = path
        self.best      = np.inf
        self.cnt       = 0
        self.best_ep   = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.cnt     = 0
            self.best_ep = epoch
            torch.save(model.state_dict(), self.path)
            return False
        self.cnt += 1
        return self.cnt >= self.patience

# ══════════════════════════════════════════════════════════
# 7. 单 epoch 训练 / 验证
# ══════════════════════════════════════════════════════════
def run_epoch(model, loader, crit, opt, device, train):
    model.train() if train else model.eval()
    tot_loss = ct = cs = n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, yt, ys in loader:
            X, yt, ys = X.to(device), yt.to(device), ys.to(device)
            pt, ps = model(X)
            loss   = crit(pt, yt) + crit(ps, ys)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot_loss += loss.item() * len(X)
            ct += (pt.argmax(1) == yt).sum().item()
            cs += (ps.argmax(1) == ys).sum().item()
            n  += len(X)
    return tot_loss / n, ct / n, cs / n

# ══════════════════════════════════════════════════════════
# 8. 绘图
# ══════════════════════════════════════════════════════════
def plot_curves(hist, save_dir):
    ep = range(1, len(hist['tl']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(ep, hist['tl'], label='Train', lw=2)
    axes[0].plot(ep, hist['vl'], label='Val',   lw=2)
    axes[0].set_title('损失曲线' if HAS_CN else 'Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, hist['tat'], label='Train', lw=2)
    axes[1].plot(ep, hist['vat'], label='Val',   lw=2)
    axes[1].set_title('故障类型准确率' if HAS_CN else 'Type Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(ep, hist['tas'], label='Train', lw=2)
    axes[2].plot(ep, hist['vas'], label='Val',   lw=2)
    axes[2].set_title('故障程度准确率' if HAS_CN else 'Severity Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0, 1.05)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[图表] 训练曲线 -> {p}')


def plot_cm(y_true, y_pred, names, title, fpath):
    cm  = confusion_matrix(y_true, y_pred)
    cmn = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, t in zip(
            axes, [cm, cmn], ['d', '.2f'],
            [title + ' (Count)', title + ' (Norm)']):
        im = ax.imshow(data, cmap='Blues', interpolation='nearest')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.set_yticklabels(names, fontsize=9)
        th = data.max() / 2
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, format(data[i, j], fmt),
                        ha='center', va='center', fontsize=8,
                        color='white' if data[i, j] > th else 'black')
        ax.set_xlabel('预测标签' if HAS_CN else 'Predicted')
        ax.set_ylabel('真实标签' if HAS_CN else 'True')
        ax.set_title(t)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[图表] 混淆矩阵 -> {fpath}')


def plot_lr(hist, save_dir):
    ep = range(1, len(hist['lr']) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(ep, hist['lr'], lw=2, color='tomato')
    plt.title('学习率曲线' if HAS_CN else 'LR Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, 'lr_curve.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[图表] 学习率曲线 -> {p}')

# ══════════════════════════════════════════════════════════
# 9. 测试集评估
# ══════════════════════════════════════════════════════════
def evaluate(model, loader, device, type_names, sev_names, title, save_dir):
    model.eval()
    all_yt, all_pt, all_ys, all_ps = [], [], [], []
    with torch.no_grad():
        for X, yt, ys in loader:
            X = X.to(device)
            pt, ps = model(X)
            all_yt.extend(yt.numpy())
            all_pt.extend(pt.argmax(1).cpu().numpy())
            all_ys.extend(ys.numpy())
            all_ps.extend(ps.argmax(1).cpu().numpy())

    all_yt = np.array(all_yt); all_pt = np.array(all_pt)
    all_ys = np.array(all_ys); all_ps = np.array(all_ps)

    acc_t = (all_yt == all_pt).mean()
    acc_s = (all_ys == all_ps).mean()

    print(f'\n{"="*65}')
    print(f'  {title}')
    print(f'{"="*65}')
    print(f'  故障类型准确率: {acc_t*100:.2f}%')
    print(f'  故障程度准确率: {acc_s*100:.2f}%')
    print('\n--- 故障类型分类报告 ---')
    print(classification_report(all_yt, all_pt,
                                target_names=type_names, digits=4))

    present  = sorted(set(all_ys.tolist()) | set(all_ps.tolist()))
    sev_lbls = [sev_names[i] for i in present]
    print('--- 故障程度分类报告 ---')
    print(classification_report(all_ys, all_ps,
                                labels=present,
                                target_names=sev_lbls, digits=4))

    tag = title.replace(' ', '_').replace('/', '_')
    plot_cm(all_yt, all_pt, type_names,
            f'{title} 故障类型' if HAS_CN else f'{title} Fault Type',
            os.path.join(save_dir, f'cm_type_{tag}.png'))
    plot_cm(all_ys, all_ps, sev_lbls,
            f'{title} 故障程度' if HAS_CN else f'{title} Severity',
            os.path.join(save_dir, f'cm_sev_{tag}.png'))
    return acc_t, acc_s

# ══════════════════════════════════════════════════════════
# 10. 主流程
# ══════════════════════════════════════════════════════════
def main():
    print('\n' + '='*65)
    print('  CWRU 轴承故障诊断  CNN-Transformer  v2（无泄露）')
    print(f'  设备: {DEVICE}')
    print('='*65)

    # ── 构建数据集 ─────────────────────────────────────────
    (X_tr, yt_tr, ys_tr,
     X_vl, yt_vl, ys_vl,
     X_st, yt_st, ys_st,
     X_ct, yt_ct, ys_ct,
     type_names, sev_names) = build_datasets(CFG)

    n_type = len(type_names)
    n_sev  = int(max(ys_tr.max(),
                     ys_vl.max(),
                     ys_st.max())) + 1

    # ── DataLoader ─────────────────────────────────────────
    def mk_loader(X, yt, ys, shuffle, noise=0.0):
        ds = BearingDS(X, yt, ys, noise_std=noise)
        return DataLoader(ds, batch_size=CFG['batch_size'],
                          shuffle=shuffle, num_workers=0, pin_memory=True)

    train_loader   = mk_loader(X_tr, yt_tr, ys_tr,
                               shuffle=True,  noise=CFG['noise_std'])
    val_loader     = mk_loader(X_vl, yt_vl, ys_vl,
                               shuffle=False, noise=0.0)
    same_loader    = mk_loader(X_st, yt_st, ys_st,
                               shuffle=False, noise=0.0)
    cross_loader   = mk_loader(X_ct, yt_ct, ys_ct,
                               shuffle=False, noise=0.0) if X_ct is not None else None

    # ── 构建模型 ───────────────────────────────────────────
    model = CNNTransformer(
        seg_len   = CFG['seg_len'],
        cnn_ch    = CFG['cnn_ch'],
        d_model   = CFG['d_model'],
        nhead     = CFG['nhead'],
        tf_layers = CFG['tf_layers'],
        dropout   = CFG['dropout'],
        n_type    = n_type,
        n_sev     = n_sev,
    ).to(DEVICE)

    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n[模型] 可训练参数: {total_p:,}')
    print(f'[模型] 故障类型数: {n_type}  故障程度数: {n_sev}')
    print(f'[模型] 类型标签: {type_names}')

    crit  = SmoothCE(CFG['label_smooth'])
    opt   = optim.AdamW(model.parameters(),
                        lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    # 使用 CosineAnnealingLR 替代 WarmRestarts
    # → 学习率单调衰减，不会产生周期性尖峰
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max   = CFG['epochs'],
        eta_min = 1e-5)

    es = EarlyStopping(
        patience  = CFG['patience'],
        min_delta = CFG['min_delta'],
        path      = os.path.join(CFG['save_dir'], 'best_model.pth'))

    hist = {'tl': [], 'vl': [],
            'tat': [], 'vat': [],
            'tas': [], 'vas': [],
            'lr': []}

    # ── 训练循环 ───────────────────────────────────────────
    print('\n' + '='*110)
    hdr = (f"{'Epoch':>6}  {'LR':>9}  "
           f"{'TrainLoss':>10}  {'ValLoss':>9}  "
           f"{'Tr-TypeAcc':>10}  {'Val-TypeAcc':>11}  "
           f"{'Tr-SevAcc':>9}  {'Val-SevAcc':>10}  "
           f"{'ES':>8}  {'Time':>6}")
    print(hdr)
    print('-'*110)

    for ep in range(1, CFG['epochs'] + 1):
        t0 = time.time()

        tl, tat, tas = run_epoch(model, train_loader, crit,
                                 opt, DEVICE, train=True)
        vl, vat, vas = run_epoch(model, val_loader,   crit,
                                 None, DEVICE, train=False)

        sched.step()
        cur_lr  = opt.param_groups[0]['lr']
        elapsed = time.time() - t0

        hist['tl'].append(tl);   hist['vl'].append(vl)
        hist['tat'].append(tat); hist['vat'].append(vat)
        hist['tas'].append(tas); hist['vas'].append(vas)
        hist['lr'].append(cur_lr)

        stop   = es.step(vl, model, ep)
        marker = ' ✓' if es.cnt == 0 else ''

        print(f"  {ep:4d}  {cur_lr:9.2e}  "
              f"{tl:10.4f}  {vl:9.4f}  "
              f"{tat*100:9.2f}%  {vat*100:10.2f}%  "
              f"{tas*100:8.2f}%  {vas*100:9.2f}%  "
              f"{es.cnt:3d}/{CFG['patience']}  "
              f"{elapsed:5.1f}s{marker}")

        if stop:
            print(f'\n[早停] 第 {ep} 轮触发  '
                  f'最优轮次={es.best_ep}  '
                  f'最优验证损失={es.best:.4f}')
            break

    print('='*110)
    print(f'\n[训练完成] 共 {ep} 轮，最优模型: {es.path}')

    # ── 加载最优权重 ───────────────────────────────────────
    model.load_state_dict(torch.load(es.path, map_location=DEVICE))

    # ── 绘图 ───────────────────────────────────────────────
    plot_curves(hist, CFG['save_dir'])
    plot_lr(hist, CFG['save_dir'])

    # ── 评估：同工况测试集 ─────────────────────────────────
    print('\n' + '█'*65)
    print('  同工况测试（训练集后15%时间段，无数据泄露）')
    print('█'*65)
    acc_t_same, acc_s_same = evaluate(
        model, same_loader, DEVICE,
        type_names, sev_names,
        '同工况测试', CFG['save_dir'])

    # ── 评估：跨工况测试集 ─────────────────────────────────
    if cross_loader is not None:
        print('\n' + '█'*65)
        print('  跨工况测试（1HP / 2HP / 3HP，模型从未见过）')
        print('█'*65)
        acc_t_cross, acc_s_cross = evaluate(
            model, cross_loader, DEVICE,
            type_names, sev_names,
            '跨工况测试', CFG['save_dir'])
    else:
        print('\n[跨工况测试] 无测试工况数据，跳过')
        acc_t_cross = acc_s_cross = None

    # ── 汇总 ───────────────────────────────────────────────
    print('\n' + '='*65)
    print('  最终结果汇总')
    print('='*65)
    print(f'  同工况 故障类型准确率: {acc_t_same*100:.2f}%')
    print(f'  同工况 故障程度准确率: {acc_s_same*100:.2f}%')
    if acc_t_cross is not None:
        print(f'  跨工况 故障类型准确率: {acc_t_cross*100:.2f}%')
        print(f'  跨工况 故障程度准确率: {acc_s_cross*100:.2f}%')
        gap_t = (acc_t_same - acc_t_cross) * 100
        gap_s = (acc_s_same - acc_s_cross) * 100
        print(f'\n  泛化差距（同工况 - 跨工况）:')
        print(f'    故障类型: {gap_t:+.2f}%  '
              f'{"（泛化良好）" if gap_t < 5 else "（存在域偏移）"}')
        print(f'    故障程度: {gap_s:+.2f}%  '
              f'{"（泛化良好）" if gap_s < 5 else "（存在域偏移）"}')
    print('='*65)
    print(f'\n[完成] 所有结果已保存至: {CFG["save_dir"]}')


if __name__ == '__main__':
    main()
